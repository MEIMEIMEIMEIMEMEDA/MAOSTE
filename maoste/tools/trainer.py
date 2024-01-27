from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer as TFSeq2SeqTrainer


class Seq2SeqWithVisionTrainer(TFSeq2SeqTrainer):

  def prediction_step(
      self,
      model: nn.Module,
      inputs: Dict[str, Union[torch.Tensor, Any]],
      prediction_loss_only: bool,
      ignore_keys: Optional[List[str]] = None,
      **gen_kwargs,
  ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    loss, generated_tokens, labels = super().prediction_step(
        model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
    return loss, generated_tokens, {
        'labels': labels,
        'img_label': inputs['img_label'],
    }
