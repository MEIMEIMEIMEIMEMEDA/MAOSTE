
import numpy as np
import torch
from transformers import EvalPrediction

from maoste.data.data_types import MetricType
from maoste.tools.tokenizer import ToknizerWrapper


def compute_span_metric_core(predictions, labels, coarse_st_index=None):
  correct_pred, total_gt, total_pred = 0., 0., 0.
  for pred_spans, label_spans in zip(predictions, labels):
    unique_pred_spans = set(map(tuple, filter(None, pred_spans)))
    unique_label_spans = set(map(tuple, filter(None, label_spans)))
    total_pred += len(unique_pred_spans)
    total_gt += len(unique_label_spans)
    if coarse_st_index is None:
      correct_pred += len(unique_pred_spans & unique_label_spans)
    else:
      unique_label_spans = list(unique_label_spans)
      index_labels = [lb[:1] for lb in unique_label_spans]
      for pred in unique_pred_spans:
        if pred[:1] not in index_labels:
          continue

        i_lb = index_labels.index(pred[:1])
        flag = 1
        for coarse_pred, coarse_label in zip(
            pred[coarse_st_index:],
            unique_label_spans[i_lb][coarse_st_index:],
        ):
          if len(set(coarse_pred) & set(coarse_label)) == 0:
            flag = 0
            break
        correct_pred += flag

  if correct_pred == 0:
    return {
        'precision': 0,
        'recall': 0,
        'f1': 0,
    }

  p = correct_pred / total_pred
  r = correct_pred / total_gt
  f1 = 2 * p * r / (p + r)
  return {
      'precision': p,
      'recall': r,
      'f1': f1,
  }


def compute_span_text_img_metric(p: EvalPrediction,
                                 tokenizer: ToknizerWrapper = None,
                                 ignore_id=-100):

  def _token_id_to_label(input_ids, img_label: torch.Tensor = None):

    def _decode(tok_ids: np.ndarray):
      vsep_index = -1
      if tokenizer.vsep_token_id in tok_ids:
        vsep_index = tok_ids.tolist().index(tokenizer.vsep_token_id)
      sentence = tokenizer.decode(
          tok_ids[:vsep_index],
          skip_special_tokens=False,
      ).replace(
          tokenizer.bos_token,
          '',
      ).replace(
          tokenizer.eos_token,
          '',
      ).replace(
          tokenizer.pad_token,
          '',
      )
      # ... | tokenizer.ssep_token | ...
      res_items = list(
          map(
              lambda x: tokenizer.label_convertor.decode(x).tolist(),
              sentence.split(tokenizer.ssep_token),
          ))
      if vsep_index != -1:
        img_region_part = tok_ids[vsep_index + 1:]
        vis_idx = 0
        for item in res_items:
          # in image
          if not item or len(item) < 3:
            continue
          if item[2] and len(img_region_part) > vis_idx and img_region_part[
              vis_idx] != tokenizer.vsep_token_id:
            item.append((img_region_part[vis_idx],))
            vis_idx += 1
          else:
            item.append((None,))
      print('res_items:', res_items)
      return res_items

    formated_res = [_decode(tok_ids) for tok_ids in input_ids]

    # Prepare label
    if img_label is not None:
      for fr, ilb in zip(formated_res, img_label):
        img_region_mask = ilb.sum(-1) != 0
        img_region_part = ilb[img_region_mask]
        vis_idx = 0
        for item in fr:
          if item[2] and len(img_region_part) > vis_idx:
            item.append(tuple(np.where(img_region_part[vis_idx] > 0)[0]))
            vis_idx += 1
          else:
            item.append((None,))
    # TODO
    return [[rr for rr in r if rr[2] != None] for r in formated_res
           ]  # entity-object-sa eval

  predictions, labels = p
  text_label = labels['labels']
  text_label[text_label == ignore_id] = tokenizer.ssep_token_id
  formated_preds = _token_id_to_label(predictions)
  formated_labels = _token_id_to_label(text_label, labels['img_label'])
  return compute_span_metric_core(formated_preds,
                                  formated_labels,
                                  coarse_st_index=-1)


def get_eval(metric_type='classification'):
  if isinstance(metric_type, str):
    metric_type = MetricType[metric_type.upper()]
  if metric_type == metric_type.SPAN_TEXT_IMG:
    return compute_span_text_img_metric
  raise ValueError(f'Please check your input {list(MetricType)}')
