from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from maoste.data.data_types import MetricType


class LossInput(NamedTuple):
  pred: torch.Tensor = None
  tgt_token_ids: torch.Tensor = None
  tgt_mask: torch.Tensor = None


class SpanLoss(nn.Module):

  def __init__(self):
    super(SpanLoss, self).__init__()

  def forward(self, loss: LossInput):
    pred, tgt, tgt_mask = loss.pred, loss.tgt_token_ids, loss.tgt_mask
    tgt = tgt.masked_fill(tgt_mask.eq(0), -100) if tgt_mask is not None else tgt
    return F.cross_entropy(
        input=pred.transpose(1, 2),
        target=tgt,
    )


def get_loss(metric_type: MetricType = MetricType.CLASSIFICATION):
  if isinstance(metric_type, str):
    metric_type = MetricType[metric_type.upper()]
  if metric_type == MetricType.CLASSIFICATION:
    return nn.CrossEntropyLoss()
  if metric_type == MetricType.SPAN:
    from maoste.tools.loss import SpanLoss
    return SpanLoss()
  raise ValueError(f'Please check your input {list(MetricType)}')