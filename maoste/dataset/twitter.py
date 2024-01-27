from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn.functional import normalize as torch_normalize
from torch.utils.data import Dataset
import yaml

from maoste.data.data_types import TwitterSample
from maoste.tools.labels import bio_to_span_indice
from maoste.tools.labels import LabelConvertorInputs
from maoste.tools.labels import words_bio_tag_to_span_text
from maoste.tools.tokenizer import TokenizerOutput
from maoste.tools.tokenizer import ToknizerWrapper

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224, 3)


@dataclass
class TwitterDatasetConifg:
  root: str
  tokenizer: ToknizerWrapper
  fields: List[str] = None
  anp_words_cfg_path: str = 'maoste/config/anp_vocab.yaml'
  iou_threshold: float = 0.5
  topk_anp_num: int = 16
  max_image_label_nums: int = 6
  vvl_region_num: int = 32
  vvl_image_feat_dim: int = 2048
  enable_vision_feature: bool = True
  enable_global_img_feat: bool = True
  caption_template: str = 'The image caption is {caption}'
  image_anp_template: str = 'The image contains {image_anp} emotional element.'


class TwitterDataset(Dataset):

  def __init__(self, config: TwitterDatasetConifg):
    self.config = config
    self.data_list = list(Path(config.root).glob('*.npz'))
    self.fields = []
    if isinstance(self.config.fields, (list, tuple)):
      self.fields = [
          f for f in self.config.fields if f in TwitterSample._fields
      ]
    self.fields = self.fields or TwitterSample._fields
    # Yang. MECSTE.
    with open(self.config.anp_words_cfg_path, 'r', encoding='utf-8') as fp:
      self._anp_words = np.array(yaml.safe_load(fp)['vocabs'])

  def __len__(self):
    return len(self.data_list)

  @property
  def tokenizer(self):
    return self.config.tokenizer

  def _to_dict(self, sample):
    # Traverse the fields and create a dictionary
    # TODO missing fields default
    return {
        field: sample['data'][i]
        for i, field in enumerate(sample['meta'])
        if field in self.fields
    }

  def _parse_aspect_iou(self, aspects: List[str],
                        ious: torch.Tensor) -> Dict[str, torch.Tensor]:
    res = {}
    if aspects is None:
      return res
    for aspect, iou in zip(aspects, ious):
      if iou.max() < self.config.iou_threshold:
        res[aspect] = None
      elif aspect in res:
        last_iou = res[aspect]
        if last_iou is None:
          res[aspect] = iou
        else:
          res[aspect] = torch.stack([last_iou, iou]).max(0)[0]
      else:
        res[aspect] = iou
    return res

  def _assign_aspect_region_labels(self,
                                   aspect_label: List[List[str]],
                                   aspect2iou: Dict[str, torch.Tensor],
                                   vvl_region_num: int = 36):
    # aspect, sentiment, is_in_image, iou
    quad_labels = []
    # object_detection_fault = {}
    for aspect, label in aspect_label:
      in_image = False
      if aspect not in aspect2iou:
        region_prob = torch.zeros(vvl_region_num)
      elif aspect2iou[aspect] is None:
        region_prob = torch.zeros(vvl_region_num)
        region_prob[-1] = 1
        in_image = True
      else:
        region_prob = aspect2iou[aspect] / aspect2iou[aspect].sum()
        in_image = True
      # quad_labels.append([aspect, label, in_image, torch.cat([region_prob, torch.tensor([not in_image])])])
      quad_labels.append([aspect, label, in_image, region_prob])
    return quad_labels

  def _get_sentence_str(self, quad_labels):
    sentences = []
    for quad in quad_labels:
      aspect, label, in_image, _ = quad
      label_name = self.tokenizer.label_name_mapping[label]
      sentence = self.tokenizer.label_convertor.encode(
          LabelConvertorInputs(
              aspect,
              label_name,
              in_image,
          ))
      sentences.append(sentence)
    return f' {self.tokenizer.ssep_token} '.join(sentences)

  def add_vision_inputs(self, sample_dict: Dict[str, Any]) -> Dict[str, Any]:
    from torchvision.ops import box_iou

    aspect2iou = None
    vvl_region_num = self.config.vvl_region_num or 32
    min_nums = min(vvl_region_num, len(sample_dict['bounding_boxes']))
    vis_attention_mask = torch.ones(
        min_nums) * self.config.enable_vision_feature
    vis_attention_mask = torch.cat(
        [vis_attention_mask,
         torch.zeros(vvl_region_num - min_nums)])
    sample_dict['box_attention_mask'] = vis_attention_mask

    vvl_boxes = torch.zeros((vvl_region_num, 4), dtype=torch.float32)
    vvl_features = torch.zeros((vvl_region_num, self.config.vvl_image_feat_dim),
                               dtype=torch.float32)
    vvl_boxes_feature_norm = torch_normalize(
        torch.tensor(sample_dict['box_features']),
        p=2,
        dim=0,
    )
    vvl_features[:min_nums] = vvl_boxes_feature_norm[:min_nums]
    _bboxes = sample_dict['bounding_boxes'][:min_nums]
    # TODO
    vvl_boxes[:min_nums] = torch.from_numpy(_bboxes) if isinstance(
        _bboxes, np.ndarray) else _bboxes
    sample_dict['box_features'] = vvl_features

    if sample_dict['boxes'] is None:
      # print(sample_dict['image_id'])
      ious = torch.zeros((1, vvl_boxes.shape[0]))
    else:
      ious = box_iou(torch.tensor(sample_dict['boxes']), vvl_boxes)
      ious[:, -1] = 0.
    ious[ious.lt(self.config.iou_threshold)] = 0.
    aspect2iou = self._parse_aspect_iou(sample_dict['aspects'], ious)
    aspect_sentiment = words_bio_tag_to_span_text(sample_dict['words'],
                                                  sample_dict['raw_target'])

    quad_label = self._assign_aspect_region_labels(
        aspect_sentiment,
        aspect2iou,
        vvl_region_num,
    )
    sample_dict['image_labels'] = torch.cat([
        torch.stack([prob for _, _, _, prob in quad_label]),
        torch.zeros(self.config.max_image_label_nums - len(quad_label),
                    vvl_region_num)
    ])

    sentence_str = self._get_sentence_str(quad_label)
    inputs = self.tokenizer.words2inputs(sentence_str, add_special_tokens=True)
    tok_output = TokenizerOutput([inputs['input_ids']],
                                 [inputs['attention_mask']])
    tok_output = tok_output.parse(
        'token_ids',
        self.tokenizer.bos_token_id,
        self.tokenizer.eos_token_id,
        self.tokenizer.max_length,
        self.tokenizer.pad_token_id,
    ).parse(
        'attention_mask',
        1,
        1,
        self.tokenizer.max_length,
        0,
    ).to_tensor('pt')
    sample_dict['decoder_token_ids'] = tok_output.token_ids[0]
    sample_dict['decoder_attention_mask'] = tok_output.attention_mask[0]
    target = tok_output.token_ids[0].clone()
    target[target == self.tokenizer.pad_token_id] = -100
    # shift left
    target[:-1] = target[1:].clone()
    sample_dict['seq_token_labels'] = target
    return sample_dict

  def caption_sentence(self, image_caption: str) -> Dict[str, Any]:
    caption_template = f' {self.tokenizer.cap_sep_token} ' + self.config.caption_template
    caption_sentence = caption_template.format(caption=image_caption.strip())
    return caption_sentence

  def image_anp_sentence(self, anp_index_arr: np.ndarray) -> Dict[str, Any]:
    anp_template = f' {self.tokenizer.img_anp_sep_token} ' + self.config.image_anp_template
    topk = self.config.topk_anp_num or 16
    topk_indices = np.argpartition(anp_index_arr, -topk)[-topk:]
    anps = ', '.join(self._anp_words[topk_indices].tolist())
    return anp_template.format(image_anp=anps.strip())

  def extend_words(self, sample_dict: Dict[str, Any],
                   sentence: str) -> Dict[str, Any]:
    word_list = sample_dict['words'].tolist()
    word_list.extend(sentence.split(' '))
    sample_dict['words'] = np.array(word_list)
    return sample_dict

  def parse_data(self, sample) -> TwitterSample:
    # TODO: convert to object
    sample_dict = self._to_dict(sample)
    skip_labels = False
    if all(
        k in sample_dict for k in ['boxes', 'bounding_boxes', 'box_features']):
      skip_labels = True
      sample_dict = self.add_vision_inputs(sample_dict)

    if 'image_caption' in sample_dict:
      sentence = self.caption_sentence(sample_dict['image_caption'])
      sample_dict = self.extend_words(sample_dict, sentence)

    if 'anp' in sample_dict and not self.config.enable_global_img_feat:
      sentence = self.image_anp_sentence(sample_dict['anp'])
      sample_dict = self.extend_words(sample_dict, sentence)

    if 'words' in sample_dict:
      # TODO: each sample one sentence
      t_ouptut = self.tokenizer.encode(
          sample_dict['words'][None,].tolist(),
          sample_dict['raw_target'][None,].tolist(),
          skip_labels=skip_labels,
      ).to_dict()
      for k, v in t_ouptut.items():
        if v is not None:
          sample_dict[k] = v[0]
    return TwitterSample(**sample_dict)

  def __getitem__(self, idx) -> TwitterSample:
    sample = np.load(self.data_list[idx], allow_pickle=True)
    return self.parse_data(sample)


class TwitterDatasetForTwoStage(TwitterDataset):

  def parse_data(self, sample) -> TwitterSample:
    sample_dict = self._to_dict(sample)
    # TODO: refactore
    sample_dict['words'] = sample_dict['words'].tolist()
    indices, img_sorted = zip(
        *sorted(enumerate(sample_dict['cropped_images']),
                key=lambda x: x[1].shape[0] * x[1].shape[1]))
    sample_dict['cropped_images'] = [
        torch.from_numpy(img) for img in img_sorted[:8]
    ]
    _span_labels = np.array(bio_to_span_indice(sample_dict['raw_target']),
                            dtype=np.int32)
    # TODO
    sample_dict['span_labels'] = np.ones((10, 3), dtype=np.int32) * -100
    sample_dict['span_labels'][:len(_span_labels), :] = _span_labels
    return TwitterSample(**sample_dict)

