from typing import List

import evaluate
import numpy as np
import torch
from transformers import EvalPrediction

from maoste.data.data_types import MetricType
from maoste.tools.tokenizer import ToknizerWrapper


def compute_classification_metric(p: EvalPrediction,
                                  tokenizer: ToknizerWrapper = None,
                                  ignore_id=-100):
  predictions, labels = p
  if len(predictions.shape) == 3:
    predictions = np.argmax(predictions, axis=-1)
  # Remove ignored index (special tokens)
  true_predictions = [[
      tokenizer.reverse_labels_mapping[p]
      for (p, l) in zip(prediction, label)
      if l != ignore_id
  ]
                      for prediction, label in zip(predictions, labels)]
  true_labels = [[
      tokenizer.reverse_labels_mapping[l] for l in label if l != ignore_id
  ] for label in labels]
  metric = evaluate.load('seqeval')
  results = metric.compute(predictions=true_predictions, references=true_labels)
  return {
      'precision': results['overall_precision'],
      'recall': results['overall_recall'],
      'f1': results['overall_f1'],
      'accuracy': results['overall_accuracy'],
      'acc_and_f1': (results['overall_f1'] + results['overall_accuracy']) / 2
  }


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
      index_labels = [lb[:coarse_st_index] for lb in unique_label_spans]
      for pred in unique_pred_spans:
        if pred[:coarse_st_index] not in index_labels:
          continue

        i_lb = index_labels.index(pred[:coarse_st_index])
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


def compute_seq_token_metric(p: EvalPrediction,
                             tokenizer: ToknizerWrapper = None,
                             ignore_id=-100):

  def _format(input_ids):
    return [
        list(
            map(
                str.strip,
                filter(
                    # lambda x: len(x) > 6 and
                    # (x.count(tokenizer.pos_token) == 1 or x.count(
                    #     tokenizer.neu_token) == 1 or x.count(tokenizer.neg_token
                    #                                         ) == 1),
                    lambda x: len(x) > 6,
                    tokenizer.decode(
                        tok_ids, skip_special_tokens=False).replace(
                            tokenizer.bos_token,
                            '').replace(tokenizer.eos_token, '').replace(
                                tokenizer.pad_token,
                                '').split(tokenizer.ssep_token))))
        for tok_ids in input_ids
    ]

  predictions, labels = p
  if len(predictions.shape) == 3:
    predictions = np.argmax(predictions, axis=-1)
  labels[labels == ignore_id] = tokenizer.ssep_token_id
  pred_words = _format(predictions)
  label_words = _format(labels)
  return compute_span_metric_core(pred_words, label_words)


def compute_span_metric(p: EvalPrediction,
                        tokenizer: ToknizerWrapper = None,
                        ignore_id=-100):
  predictions, labels = p
  step = 3

  def _format_pred(pred):
    spans = []
    # next of bos
    for i in range(1, len(pred), step):
      if (pred[i] >= len(pred) or \
          pred[i] == tokenizer.pad_token_id or \
          pred[i] == tokenizer.eos_token_id):
        continue
      span = pred[i:i + step].tolist()
      if len(span) < 3:
        continue
      spans.append(span)
    return set(map(tuple, spans))

  def _format_label(label):
    spans = []
    for i in range(0, len(label), step):
      if (label[i] >= len(label) or \
          label[i] == ignore_id or \
          label[i] == tokenizer.pad_token_id or \
          label[i] == tokenizer.eos_token_id):
        continue
      span = label[i:i + step].tolist()
      if len(span) < 3:
        continue
      spans.append(span)
    return set(map(tuple, spans))

  pred_spans = []
  label_spans = []
  for prediction, label in zip(predictions, labels):
    pred_spans.append(_format_pred(prediction))
    label_spans.append(_format_label(label))

  return compute_span_metric_core(pred_spans, label_spans)


def compute_two_stage_metric(p: EvalPrediction,
                             tokenizer: ToknizerWrapper = None,
                             ignore_id=-100):
  predictions, labels = p
  pred_spans = []
  label_spans = []
  for prediction, label in zip(predictions, labels):
    pred_spans.append([(int(st), int(ed), int(polarity))
                       for st, ed, polarity in prediction
                       if st != ignore_id and ed != ignore_id])
    label_spans.append([(int(st), int(ed), int(polarity))
                        for st, ed, polarity in label
                        if st != ignore_id and ed != ignore_id])
  return compute_span_metric_core(pred_spans, label_spans)


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
            # item.append((img_region_part[vis_idx].argmax(),))
            item.append(tuple(np.where(img_region_part[vis_idx] > 0)[0]))
            vis_idx += 1
          else:
            item.append((None,))
    # TODO
    return [[rr for rr in r if rr[2] != None] for r in formated_res
           ]  # entity-object-sa eval
    # return [[tuple(rr[:2]) for rr in r] for r in formated_res] # sa eval
    # return [[tuple(rr[:2]) for rr in r if rr[2] != None] for r in formated_res]  # sa eval for region is not none

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
  if metric_type == MetricType.CLASSIFICATION:
    return compute_classification_metric
  if metric_type == MetricType.SEQ_TOKEN:
    return compute_seq_token_metric
  if metric_type == MetricType.SPAN:
    return compute_span_metric
  if metric_type == MetricType.SPAN_TWO_STAGE:
    return compute_two_stage_metric
  if metric_type == metric_type.SPAN_TEXT_IMG:
    return compute_span_text_img_metric
  raise ValueError(f'Please check your input {list(MetricType)}')
