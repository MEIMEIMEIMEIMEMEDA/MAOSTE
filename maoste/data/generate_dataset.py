from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import BertTokenizer
import yaml

from maoste.data.data_types import AspectSentiment
from maoste.data.data_types import ImageAspectBBox
from maoste.data.data_types import ImageVinVL
from maoste.data.data_types import RegionAnp
from maoste.data.data_types import SentenceAspect
from maoste.data.data_types import TwitterSample

logger = logging.getLogger(__name__)
WORKER_NUMS = max(1, os.cpu_count() // 2)

with open('maoste/config/anp_vocab.yaml', 'r', encoding='utf-8') as _fp:
  ANP_VOCAB_CFG = yaml.safe_load(_fp)['vocabs']

# TODO: region anp


def _anp_to_array(anp: dict):
  return np.array([anp[k] for k in ANP_VOCAB_CFG])


def _proto_tensor_to_numpy_tensor(tensor: Tensor,
                                  data_type) -> Optional[np.ndarray]:
  if tensor.nelement() == 0:
    return None
  np_tensor = np.array(tensor.data, dtype=data_type)
  np_tensor = np_tensor.reshape(tensor.shape)
  return np_tensor


def _convert_tensor_to_numpy(dict_data: Dict[str, Any]):
  for k, v in dict_data.items():
    if isinstance(v, Tensor):
      dict_data[k] = _proto_tensor_to_numpy_tensor(v, np.float32)
  return dict_data


class _TwitterSample(NamedTuple):
  image_id: str = None
  image: np.ndarray = None
  cropped_images: List[np.ndarray] = None
  vinvl: ImageVinVL = None  # .npz
  image_caption: str = None
  anp: Union[Tensor, np.ndarray] = None
  region_anp: Union[Tensor, np.ndarray] = None
  aspect_bbox: ImageAspectBBox = None  # xml
  aspect_sentiment: AspectSentiment = None  # txt

  def name_tuple_fields(self):
    """Flatten name tuple fields."""
    return ['vinvl', 'aspect_bbox', 'aspect_sentiment']

  def to_dict(self) -> Dict[str, np.ndarray]:
    data = {}
    for f in _TwitterSample._fields:
      value = getattr(self, f)
      if value is None:
        continue
      if f in self.name_tuple_fields():
        data.update(value._asdict())
      else:
        data[f] = value
    return _convert_tensor_to_numpy(data)


@dataclass
class RawDataConfig:
  src_root: Path
  dst_root: Path
  image_root: Path = None
  cropped_image_root: Path = None
  xml_root: Path = None
  vinvl_root: Path = None
  anp_fn: Path = None
  region_anp_vocab_fn: Path = None
  caption_fn: Path = None

  def __post_init__(self):
    self.src_root = Path(self.src_root)
    self.dst_root = Path(self.dst_root)
    if not self.image_root:
      self.image_root = self.src_root / 'image'
    if not self.cropped_image_root:
      self.cropped_image_root = self.src_root / 'cropped_images'
    if not self.xml_root:
      self.xml_root = self.src_root / 'xml'
    if not self.vinvl_root:
      self.vinvl_root = self.src_root / 'image_VinVL_Object_36'
    if not self.anp_fn:
      self.anp_fn = self.src_root / 'image_anp_output.json'
    if not self.caption_fn:
      self.caption_fn = self.src_root / 'image_caption_output.json'
    if not self.region_anp_vocab_fn:
      self.region_anp_vocab_fn = self.src_root / 'region_anp_vocab.yaml'
      if not self.region_anp_vocab_fn.exists():
        img_region_fns = self.cropped_image_root.glob('**/region_data.json')
        all_region_anp_vocabs = set()
        for fn in tqdm(img_region_fns, 'Init region anp vocab yaml'):
          with open(fn, 'r', encoding='utf-8') as _fp:
            res = json.load(_fp)
            all_region_anp_vocabs |= set(
                [anp for v in res.values() for anp in v])
        with open(self.region_anp_vocab_fn, 'w', encoding='utf-8') as _fp:
          yaml.dump({'vocabs': sorted(list(all_region_anp_vocabs))}, _fp)


class RawDataProcessor:

  def __init__(self, cfg: RawDataConfig) -> None:
    self._cfg = cfg
    self._asp_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(self._cfg.anp_fn, 'r', encoding='utf8') as fp:
      self._image2anp = json.load(fp)
    with open(self._cfg.caption_fn, 'r', encoding='utf-8') as fp:
      self._image_caption_mapping = json.load(fp)
    with open(self._cfg.region_anp_vocab_fn, 'r', encoding='utf-8') as fp:
      self._region_anp_vocab = yaml.safe_load(fp)['vocabs']

  def _get_img(self, img_id) -> np.ndarray:
    image_path = self._cfg.image_root / f'{img_id}.jpg'
    return np.asarray(Image.open(image_path))

  def _get_cropped_imgs(self, img_id) -> List[np.ndarray]:
    crop_img_paths = (self._cfg.cropped_image_root / img_id).glob('*.jpg')
    crop_images = []
    for fn in sorted(crop_img_paths, key=lambda x: int(x.stem.split('_')[-1])):
      crop_images.append(np.asarray(Image.open(fn)))
    return crop_images

  def _get_vinvl(self, img_id) -> ImageVinVL:
    vinvl_npz_path = self._cfg.vinvl_root / f'{img_id}.jpg.npz'
    return ImageVinVL.from_npz_file(vinvl_npz_path)

  def _get_img_caption(self, img_id: str) -> str:
    img_filename = img_id if img_id.endswith('.jpg') else img_id + '.jpg'
    if img_filename in self._image_caption_mapping:
      return self._image_caption_mapping[img_filename]
    return ''

  def _get_anp(self, img_id) -> np.ndarray:
    return _anp_to_array(self._image2anp[f'{img_id}.jpg'])

  def _get_region_anp(self, img_id) -> np.ndarray:
    region_anp_cfg = self._cfg.cropped_image_root / img_id / 'region_data.json'
    region_anps = RegionAnp.from_json(region_anp_cfg)
    region_anps_arr = np.zeros((len(region_anps), len(self._region_anp_vocab)))
    for i, region_anp_dict in enumerate(region_anps):
      for k, v in region_anp_dict.items():
        region_anps_arr[i, self._region_anp_vocab.index(k)] = v
    return region_anps_arr

  def _get_aspect_bbox(self, img_id) -> ImageAspectBBox:
    return ImageAspectBBox.from_xml_file(self._cfg.xml_root / f'{img_id}.xml')

  def _get_aspect_sentiment(self, row) -> AspectSentiment:
    return AspectSentiment.from_dict(row)

  def parse_data(self, row) -> TwitterSample:
    image_id = row['img_id']
    _sample = _TwitterSample(
        image_id,
        self._get_img(image_id),
        self._get_cropped_imgs(image_id),
        self._get_vinvl(image_id),
        self._get_img_caption(image_id),
        self._get_anp(image_id),
        self._get_region_anp(image_id),
        self._get_aspect_bbox(image_id),
        self._get_aspect_sentiment(row),
    )
    return TwitterSample(**_sample.to_dict())


def _parse_and_save(processor: RawDataProcessor, dataset_dst, item):
  data = processor.parse_data(item)
  fn = dataset_dst / f'{item["img_id"]}.npz'
  with open(fn, 'wb') as fp:
    np.savez_compressed(fp,
                        data=np.array(data, dtype=object),
                        meta=TwitterSample._fields)
    logger.info('Save to %s' % fn)


def convert_to_npz(processor: RawDataProcessor, mode='train'):
  dataset_dst = processor._cfg.dst_root / mode
  dataset_dst.mkdir(parents=True, exist_ok=True)
  start = time.time()
  df = SentenceAspect.load_df_from_files(processor._cfg.src_root /
                                         f'{mode}.txt')
  pbar = tqdm(df.iterrows(), desc=mode, total=len(df))
  # Create a ThreadPoolExecutor with worker threads
  with ThreadPoolExecutor(max_workers=WORKER_NUMS) as executor:
    futures = [
        executor.submit(_parse_and_save, processor, dataset_dst, item)
        for _, item in pbar
    ]
  # Wait for all threads to complete
  for future in futures:
    future.result()
  print(f'Runtime: {time.time() - start:.2f} seconds, data:')


if __name__ == '__main__':
  raw_data_cfg = RawDataConfig(
      src_root='/original_dataset/finish_processed_data/',
      dst_root='/original_dataset/finish_processed_data/npz',
  )
  processor = RawDataProcessor(raw_data_cfg)
  convert_to_npz(processor, 'dev')
  convert_to_npz(processor, 'test')
  convert_to_npz(processor, 'train')
