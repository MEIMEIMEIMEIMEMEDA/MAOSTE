from enum import Enum
import json
import os
from typing import Dict, List, NamedTuple, Union

import numpy as np
import pandas as pd
from torch import Tensor


class ModuleType(Enum):
  NONE = 0
  CRF = 1


class TaskType(Enum):
  TEXT_TOKEN_CLS = 0
  TEXT_SEQ2SEQ = 1
  MM_TWO_STAGE = 2
  MM_GEN = 3


class TrainerType(Enum):
  COMMON = 0
  SEQ2SEQ = 1


class LabelType(Enum):
  BIO = 0
  SENTIMENT = 1
  BIO_SENTIMENT = 2


class LabelConvertorType(Enum):
  ASPECT_LABEL_EXIST = 0
  ASPECT_LABEL_EXIST_WITH_POLARITY = 1
  ASPECT_EXIST_LABEL = 2
  ASPECT_EXIST_LABEL_WITH_POLARITY = 3


class MetricType(Enum):
  CLASSIFICATION = 0
  SEQ_TOKEN = 1
  SPAN = 2
  SPAN_TWO_STAGE = 3
  SPAN_TEXT_IMG = 4


class RegionAnp:

  @classmethod
  def custom_sort_key(cls, item):
    file_name = item[0]
    parts = file_name.split('_')
    last_part = parts[-1]
    digits = ''.join(filter(str.isdigit, last_part))
    last_digit = int(digits) if digits else 0
    return last_digit

  @classmethod
  def from_json(cls, fn: str) -> List[Dict[str, float]]:
    if not os.path.exists(fn):
      return None
    with open(fn, 'r') as fp:
      data = json.load(fp)
    sorted_data = sorted(data.items(), key=RegionAnp.custom_sort_key)
    second_elements = [item[1] for item in sorted_data]
    return second_elements


class ImageAspectBBox(NamedTuple):
  aspects: Union[Tensor, np.ndarray] = None
  boxes: Union[Tensor, np.ndarray] = None

  @classmethod
  def from_xml_file(cls, fn: str):
    import xml.etree.ElementTree as ET
    if not os.path.exists(fn):
      return None
    fn = str(fn)
    tree = ET.parse(fn)
    root = tree.getroot()
    aspects = []
    boxes = []
    for object_container in root.findall('object'):
      for names in object_container.findall('name'):
        box_name = names.text
        box_container = object_container.findall('bndbox')
        if len(box_container) > 0:
          xmin = int(box_container[0].findall('xmin')[0].text)
          ymin = int(box_container[0].findall('ymin')[0].text)
          xmax = int(box_container[0].findall('xmax')[0].text)
          ymax = int(box_container[0].findall('ymax')[0].text)
        aspects.append(box_name)
        boxes.append([xmin, ymin, xmax, ymax])
    return cls(np.array(aspects), np.array(boxes))


class ImageVinVL(NamedTuple):
  num_boxes: Union[Tensor, np.ndarray] = None
  image_h: Union[Tensor, np.ndarray] = None
  image_w: Union[Tensor, np.ndarray] = None
  # bbox (N,4)
  bounding_boxes: Union[Tensor, np.ndarray] = None
  # bbox vinvl feature, (N, feat_dim)
  box_features: Union[Tensor, np.ndarray] = None
  # bbox score (N,)
  scores: Union[Tensor, np.ndarray] = None
  # object type str (N,), e.g. 'sky'
  objects: Union[Tensor, np.ndarray] = None
  # object adj and noun str (N,), e.g. 'blue sky'
  attr_obj: Union[Tensor, np.ndarray] = None
  # object attr score (N, 1)
  attr_scores: Union[Tensor, np.ndarray] = None
  # object attr score distribution (N, m)
  scores_all: Union[Tensor, np.ndarray] = None

  @classmethod
  def from_npz_file(cls, npz_file: str):
    if os.path.exists(npz_file):
      input_kwargs = {
          k: v for k, v in np.load(npz_file, allow_pickle=True).items()
      }
      return cls(**input_kwargs)


class AspectSentiment(NamedTuple):
  words: Union[Tensor, np.ndarray] = None
  word_ids: Union[Tensor, np.ndarray] = None
  tokens: Union[Tensor, np.ndarray] = None
  token_ids: np.ndarray = None
  token_type_ids: np.ndarray = None
  attention_mask: np.ndarray = None
  token_labels: np.ndarray = None
  span_labels: Union[Tensor, np.ndarray] = None
  raw_target: Union[Tensor, np.ndarray] = None

  @classmethod
  def from_dict(cls, data):
    input_kwargs = {}
    for f in cls._fields:
      if f in data and data[f] is not None:
        input_kwargs[f] = np.asarray(data[f])
    if input_kwargs:
      return cls(**input_kwargs)


def read_txt_file(text_file):
  f = open(text_file)
  data = []
  raw_data = []
  target = []
  for line in f:
    if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == '\n':
      if len(raw_data) > 0:
        data.append((raw_data, target))
        raw_data = []
        target = []
      continue
    splits = line.split('\t')
    if len(splits) == 1:
      raw_data.append(splits[0][:-1])  # img id
    else:
      raw_data.append(splits[0])  # text
      target.append(splits[-1][:-1])  # target
  if len(raw_data) > 0:
    data.append((raw_data, target))
    raw_data = []
    target = []
  print('The number of samples: ' + str(len(data)))
  return data


class SentenceAspect:

  @classmethod
  def load_df_from_files(cls, text_file) -> pd.DataFrame:
    items = []
    text_file_data = read_txt_file(text_file)
    for raw_words, target in text_file_data:
      img_id = raw_words[0][6:]
      raw_words = raw_words[1:]
      item = {
          'img_id': img_id,
          'words': raw_words,
          'raw_target': target,
      }
      items.append(item)
    return pd.DataFrame(items)


class TwitterSample(NamedTuple):
  image_id: str = None
  image: np.ndarray = None
  cropped_images: List[np.ndarray] = None
  # ========= vinvl =============
  num_boxes: Union[Tensor, np.ndarray] = None
  image_h: Union[Tensor, np.ndarray] = None
  image_w: Union[Tensor, np.ndarray] = None
  bounding_boxes: Union[Tensor, np.ndarray] = None
  box_features: Union[Tensor, np.ndarray] = None
  region_anp: Union[Tensor, np.ndarray] = None
  # bbox score (N,)
  scores: Union[Tensor, np.ndarray] = None
  # object type str (N,), e.g. 'sky'
  objects: Union[Tensor, np.ndarray] = None
  # object adj and noun str (N,), e.g. 'blue sky'
  attr_obj: Union[Tensor, np.ndarray] = None
  # object attr score (N, 1)
  attr_scores: Union[Tensor, np.ndarray] = None
  # object attr score distribution (N, m)
  scores_all: Union[Tensor, np.ndarray] = None
  # ========= image caption =============
  image_caption: str = None
  # ========= anp =============
  anp: Union[Tensor, np.ndarray] = None
  # ========= Image Aspect Sentiment =============
  words: Union[Tensor, np.ndarray] = None
  word_ids: Union[Tensor, np.ndarray] = None
  # ========= Encoder Input ========
  tokens: Union[Tensor, np.ndarray] = None
  token_ids: np.ndarray = None
  token_type_ids: np.ndarray = None
  attention_mask: np.ndarray = None
  # ========= Region Aspect Sentiment =============
  aspects: Union[Tensor, np.ndarray] = None
  boxes: Union[Tensor, np.ndarray] = None
  # ========= Decoder Input for Seq2Seq Mdoel ========
  decoder_token_ids: Union[Tensor, np.ndarray] = None
  decoder_attention_mask: Union[Tensor, np.ndarray] = None
  # ========== Labels =========
  token_labels: np.ndarray = None
  # (M, 2)
  seq_token_labels: Union[Tensor, np.ndarray] = None
  # (M, 2)
  span_labels: Union[Tensor, np.ndarray] = None
  raw_target: Union[Tensor, np.ndarray] = None
  box_attention_mask: Union[Tensor, np.ndarray] = None
  image_labels: Union[Tensor, np.ndarray] = None
