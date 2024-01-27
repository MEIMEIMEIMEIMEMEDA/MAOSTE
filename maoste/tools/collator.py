from collections import defaultdict
import logging
from typing import Dict, List

import numpy as np
import torch

from maoste.data.data_types import TaskType
from maoste.data.data_types import TwitterSample

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = [
    np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64,
    np.int32, np.int16, np.int8, np.uint8, np.bool_
]


def _collate(batch: List[TwitterSample],
             need_fields: List[str] = None,
             is_stack=True) -> Dict[str, torch.Tensor]:
  """Custom collate function to collate a list of TwitterSample instances into
  a single TwitterSample instance."""
  collated_batch = defaultdict(list)
  # Stack or concatenate the data for each field
  for i, f in enumerate(TwitterSample._fields):
    if need_fields is not None and f not in need_fields:
      continue
    msg = f'Collate field {f}'
    logger.info(msg)

    for item in batch:
      if isinstance(item[i], np.ndarray) and item[i].dtype in SUPPORTED_DTYPES:
        collated_batch[f].append(torch.from_numpy(item[i]))
      elif item[i] is not None:
        collated_batch[f].append(item[i])
    if f not in collated_batch:
      continue
    if is_stack and isinstance(collated_batch[f][0], torch.Tensor):
      collated_batch[f] = torch.stack(collated_batch[f])

  # Post check
  missing_fields = []
  for sub_field in need_fields:
    if sub_field not in collated_batch:
      missing_fields.append(sub_field)
  return collated_batch


def collate_mm_generation_data(batch: List[TwitterSample]):
  collated_batch = _collate(batch, [
      'image_id',
      'anp',
      'token_ids',
      'attention_mask',
      'box_features',
      'box_attention_mask',
      'decoder_token_ids',
      'decoder_attention_mask',
      'image_labels',
      'seq_token_labels',
  ], True)

  return {
      'input_ids':
          collated_batch['token_ids'].to(torch.long),
      'attention_mask':
          collated_batch['attention_mask'].to(torch.bool),
      'vis_feats':
          collated_batch['box_features'],
      'vis_attention_mask':
          collated_batch['box_attention_mask'].to(torch.bool),
      'decoder_input_ids':
          collated_batch['decoder_token_ids'].to(torch.long),
      'decoder_attention_mask':
          collated_batch['decoder_attention_mask'].to(torch.bool),
      'img_label':
          collated_batch['image_labels'].to(torch.float32),
      'img_anp_label':
          collated_batch['anp'].to(torch.float32) if isinstance(
              collated_batch['anp'], torch.Tensor) else torch.empty(0),
      'img_id':
          collated_batch['image_id'],
      'labels':
          collated_batch['seq_token_labels'].to(torch.long),
  }


def get_collator(task_type='TEXT_TOKEN_CLS'):
  if isinstance(task_type, str):
    task_type = TaskType[task_type.upper()]
  if task_type == TaskType.MM_GEN:
    return collate_mm_generation_data
  raise ValueError(f'Please check your input {list(TaskType)}')
