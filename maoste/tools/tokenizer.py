from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from itertools import chain
from typing import Dict, List, NamedTuple, Union

import numpy as np
from torch import Tensor
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer
from transformers.utils import TensorType

from maoste.data.data_types import LabelConvertorType
from maoste.tools.labels import get_label_convertor
from maoste.tools.labels import LabelConvertor


@dataclass
class TokenizerConfig:
  label_indice_mapping: Dict[str, int]
  reverse_labels_mapping: List[int]
  label_convertor_type: LabelConvertorType = LabelConvertorType.ASPECT_LABEL_EXIST
  label_name_mapping: Dict[str, str] = field(default_factory=lambda: {
      'POS': 'positive',
      'NEG': 'negative',
      'NEU': 'neutral'
  })
  padding_max_len: int = 80


@dataclass
class TokenizerOutput:
  token_ids: Union[Tensor, np.ndarray, List] = None
  attention_mask: Union[Tensor, np.ndarray, List] = None
  word_ids: Union[Tensor, np.ndarray, List] = None
  decoder_token_ids: Union[Tensor, np.ndarray, List] = None
  decoder_attention_mask: Union[Tensor, np.ndarray, List] = None
  token_labels: Union[Tensor, np.ndarray, List] = None
  seq_token_labels: Union[Tensor, np.ndarray, List] = None
  span_labels: Union[Tensor, np.ndarray, List] = None

  def to_dict(self):
    return asdict(self)

  @property
  def skip_fields(self):
    return ['word_ids']

  def to_tensor(self, rt_type='np'):
    if rt_type == TensorType.PYTORCH:

      def as_tensor(value, dtype=None):
        if value is None:
          return None
        return Tensor(value)

    elif rt_type == TensorType.NUMPY:

      def as_tensor(value, dtype=None):
        if value is None:
          return None
        value_lens = [len(val) for val in value]
        if len(set(value_lens)) > 1 and dtype is None:
          # we have a ragged list so handle explicitly
          value = as_tensor([np.asarray(val) for val in value], dtype=object)
        return np.asarray(value, dtype=dtype)

    for f in fields(self):
      if f.name in self.skip_fields:
        continue
      setattr(self, f.name, as_tensor(getattr(self, f.name, None)))
    return self

  def build_with_special_token(self, field: str, st, ed):
    ids = getattr(self, field, None)
    if ids is None:
      return self

    st = st if isinstance(st, list) else [st]
    ed = ed if isinstance(ed, list) else [ed]
    setattr(self, field, [st + i + ed for i in ids])
    return self

  def pad(self, field: str, max_length: int, pad_id: int):
    ids = getattr(self, field, None)
    if ids is None:
      return self

    for i, require_input in enumerate(ids):
      difference = max_length - len(require_input)
      if difference <= 0:
        continue
      ids[i] = require_input + [pad_id] * difference

    setattr(self, field, [i[:max_length] for i in ids])
    return self

  def parse(self, field, st, ed, max_length, pad_id):
    self.build_with_special_token(field, st, ed).pad(field, max_length, pad_id)
    return self


class SpecialTokens(NamedTuple):
  bos_token: str = '<s>'
  eos_token: str = '</s>'
  sep_token: str = '<sep>'
  cls_token: str = '<s>'
  unk_token: str = '<unk>'
  pad_token: str = '<pad>'
  mask_token: str = '<mask>'
  additional_special_tokens: Dict[str, str] = {
      'begin_token': '<B>',
      'inner_token': '<I>',
      'out_token': '<O>',
      'pos_token': '<POS>',
      'neg_token': '<NEG>',
      'neu_token': '<NEU>',
      'ssep_token': '<ssep>',
      'vsep_token': '<vsep>',
      'cap_sep_token': '<csep>',
      'img_anp_sep_token': '<iasep>',
      'img_region_anp_sep_token': '<irasep>',
  }


class ToknizerWrapper:

  def __init__(
      self,
      tokenizer: PreTrainedTokenizer,
      config: TokenizerConfig,
  ):
    self.config = config
    self._tokenizer = tokenizer
    self._span_target_ids = []
    # NOTE: ensure same token of each experiment
    self.add_special_tokens(SpecialTokens()._asdict())

  def __len__(self):
    return len(self._tokenizer)

  def __setattr__(self, key, value):
    super().__setattr__(key, value)

  def __getattribute__(self, key):
    return super().__getattribute__(key)

  @property
  def reverse_labels_mapping(self):
    return self.config.reverse_labels_mapping

  @property
  def vocab_size(self):
    return self._tokenizer.vocab_size

  @property
  def span_targets(self):
    return [
        '<s>', '<pad>', '</s>', '<B>', '<I>', '<O>', '<POS>', '<NEG>', '<NEU>'
    ]

  @property
  def label_indice_mapping(self):
    return self.config.label_indice_mapping

  @property
  def label_name_mapping(self):
    return self.config.label_name_mapping

  @property
  def span_target_ids(self) -> List[int]:
    if len(self._span_target_ids) == len(self.span_targets):
      return self._span_target_ids
    self._span_target_ids = self._tokenizer.convert_tokens_to_ids(
        self.span_targets)
    assert len(self._span_target_ids) == len(
        self.span_targets), 'Please add target by self.add_special_tokens()'
    return self._span_target_ids

  @property
  def max_length(self):
    return self.config.padding_max_len

  @property
  def label_convertor(self) -> LabelConvertor:
    return get_label_convertor(self.config.label_convertor_type)

  def add_special_tokens(self, new_tokens: Dict[str, str]):
    add_special_tokens = new_tokens.pop('additional_special_tokens')
    sepcial_tokens = new_tokens.copy()

    # Add special token
    _asp_values = list(add_special_tokens.values())
    new_tokens['additional_special_tokens'] = _asp_values
    self._tokenizer.add_special_tokens(new_tokens)
    if getattr(self._tokenizer, 'unique_no_split_tokens', None):
      self._tokenizer.unique_no_split_tokens += _asp_values
    # Reset additional special token
    for k, v in add_special_tokens.items():
      tk_id = self._tokenizer.convert_tokens_to_ids(v)
      setattr(self._tokenizer, f'{k}', v)
      setattr(self._tokenizer, f'{k}_id', tk_id)
      setattr(self, f'{k}', v)
      setattr(self, f'{k}_id', tk_id)

    # Reset special token
    for k, v in sepcial_tokens.items():
      tk_id = self._tokenizer.convert_tokens_to_ids(v)
      setattr(self._tokenizer, f'{k}', v)
      setattr(self._tokenizer, f'{k}_id', tk_id)
      setattr(self, f'{k}', v)
      setattr(self, f'{k}_id', tk_id)

  def add_tokens(self, new_tokens: Dict[str, str]):
    self._tokenizer.add_tokens(new_tokens)

  def _parse(self,
             tok_output: TokenizerOutput,
             rt_type='np') -> TokenizerOutput:
    tok_output.parse(
        'token_ids',
        self.bos_token_id,
        self.eos_token_id,
        self.max_length,
        self.pad_token_id,
    ).parse(
        'attention_mask',
        1,
        1,
        self.max_length,
        0,
    ).parse(
        'word_ids',
        None,
        None,
        self.max_length,
        None,
    ).parse(
        'decoder_token_ids',
        self.bos_token_id,
        self.eos_token_id,
        self.max_length,
        self.pad_token_id,
    ).parse(
        'decoder_attention_mask',
        1,
        1,
        self.max_length,
        0,
    ).parse(
        'token_labels',
        -100,  # torch ignore label
        -100,
        self.max_length,
        -100,
    ).parse(
        'seq_token_labels',
        self.bos_token_id,  # torch ignore label
        self.eos_token_id,
        self.max_length,
        -100,
    ).parse(
        'span_labels',
        [],
        self.span_targets.index(self.eos_token),
        self.max_length,
        -100,
    ).to_tensor(rt_type)
    return tok_output

  def words2inputs(self, words, add_special_tokens=False) -> BatchEncoding:
    tokenized_inputs = self._tokenizer(
        words,
        is_split_into_words=not isinstance(words[0], str),
        add_special_tokens=add_special_tokens,
    )
    return tokenized_inputs

  def get_input_word_ids(self,
                         tokenized_inputs: BatchEncoding) -> List[List[int]]:
    return [
        tokenized_inputs.word_ids(batch_index=i)
        for i in range(len(tokenized_inputs['input_ids']))
    ]

  def get_input_token_ids(self, tokenized_inputs: BatchEncoding):
    return np.asarray([
        self._tokenizer.convert_ids_to_tokens(token_ids)
        for token_ids in tokenized_inputs['input_ids']
    ])

  def token_label_to_span(self,
                          token_label_ids: np.ndarray,
                          reverse_label_mapping=None,
                          offset: int = None) -> np.ndarray:
    res = []
    _rlm = reverse_label_mapping or self.reverse_labels_mapping
    _offset = offset if isinstance(offset, int) else len(self.span_targets)
    for i, tk_label_id in enumerate(token_label_ids):
      if tk_label_id <= 0:
        continue
      lb_name = f'<{_rlm[tk_label_id][-3:]}>'
      # shift, span index start from size of self.span_targets
      st = i + _offset
      # Init res triple elem when meet B-xxx
      if tk_label_id % 2 == 1:
        res.append([st, st, self.span_targets.index(lb_name)])
      elif len(res) == 0:
        res.append([st - 1, st, self.span_targets.index(lb_name)])
      else:
        # increase ed
        res[-1][1] += 1
    return res

  def word_labels_align_token_labels(self, word_ids: List[int],
                                     word_labels: List[str]) -> List[List[int]]:
    # ignore default: -100
    token_labels = np.ones(len(word_ids)) * -100
    pre_wid = None
    for j, wid in enumerate(word_ids):
      if wid is None or word_labels[wid] is None:
        continue
      if pre_wid == wid:
        token_labels[j] = self.label_indice_mapping[word_labels[wid]]
        # if B word is splited, which subpart label is I
        if 'B' in word_labels[wid]:
          token_labels[j] += 1
      else:
        token_labels[j] = self.label_indice_mapping[word_labels[wid]]
        pre_wid = wid
    return token_labels.astype(dtype=np.int32).tolist()

  def word_label_to_seq_token(self, words, words_labels) -> List[int]:
    seq_words = []
    polarity = None
    for word, word_label in zip(words, words_labels):
      if 'O' == word_label:
        continue
      if 'B' in word_label:
        if polarity is not None:
          seq_words += [' ', polarity, ' ', self.ssep_token, ' ']
        polarity = f'is {self.config.label_name_mapping[word_label[-3:]]}'
      seq_words.append(word)
    seq_words += [' ', polarity]
    return seq_words

  def token_masks_to_word_masks(self, word_ids, token_mask) -> List[List[int]]:
    # mask default: 0
    word_masks = []
    for j, wid in enumerate(word_ids):
      if wid is None:
        continue
      if token_mask[j] == 0:
        continue
      if len(word_masks) < wid + 1:
        word_masks.append(0)
      if word_masks[wid] != 0:
        # NOTE: increase recall
        continue
      word_masks[wid] = token_mask[j]

    return word_masks

  def word_masks_to_token_masks(self, word_ids, word_masks) -> List[List[int]]:
    # mask default: 0
    token_masks = np.zeros(len(word_ids) - 1)
    for j, wid in enumerate(word_ids[1:]):
      if wid is None:
        break
      token_masks[j] = word_masks[wid]
    return [1] + token_masks.astype(dtype=np.int32).tolist()

  def gen_labels(self, input_words: List[str],
                 input_token_word_ids: List[List[int]],
                 words_labels: List[List[str]]) -> Dict[str, np.ndarray]:
    token_labels = []
    seq_token_labels = []
    span_labels = []
    for words, word_ids, word_labels in zip(input_words, input_token_word_ids,
                                            words_labels):
      token_labels.append(
          self.word_labels_align_token_labels(
              word_ids,
              word_labels,
          ))
      seq_token_labels.append(self.word_label_to_seq_token(words, word_labels))
      span_labels.append(self.token_label_to_span(token_labels[-1]))
    seq_token_labels = [
        tok_ids
        for tok_ids in self.words2inputs(seq_token_labels, True)['input_ids']
    ]
    return {
        'token_labels': token_labels,
        'seq_token_labels': seq_token_labels,
        'span_labels': span_labels,
    }

  def gen_decoder_inputs(self, encode_input_ids: List[np.ndarray],
                         labels: List[np.ndarray]):
    decoder_input_ids = []
    attention_mask = []
    for _, label in zip(encode_input_ids, labels):
      # TODO
      decoder_input_ids.append(label)
      attention_mask.append(np.ones_like(label).tolist())
    return {'input_ids': decoder_input_ids, 'attention_mask': attention_mask}

  def encode(self,
             words: List[List[str]],
             word_labels: List[List[str]],
             skip_labels: bool = False,
             rt_type='np') -> TokenizerOutput:
    # =========== Prepare inputs and labels ===============
    inputs = self.words2inputs(words)
    word_ids = self.get_input_word_ids(inputs)
    if not skip_labels:
      labels = self.gen_labels(words, word_ids, word_labels)
      decoder_inputs = self.gen_decoder_inputs(inputs['input_ids'],
                                               labels['seq_token_labels'])
      # =========== Post parse output data ===============
      tok_output = TokenizerOutput(
          inputs['input_ids'],
          inputs['attention_mask'],
          word_ids,
          decoder_inputs['input_ids'],
          decoder_inputs['attention_mask'],
          labels['token_labels'],
          labels['seq_token_labels'],
          [list(chain(*lb)) for lb in labels['span_labels']],
      )
    else:
      tok_output = TokenizerOutput(
          inputs['input_ids'],
          inputs['attention_mask'],
          word_ids,
      )
    tok_output = self._parse(tok_output, rt_type)
    return tok_output

  def decode(self, input_ids, **kwargs):
    return self._tokenizer.decode(input_ids, **kwargs)

  def decode_from_output(self, tok_output: TokenizerOutput):
    return [(self.decode(word_tok_ids, skip_special_tokens=True),
             self.decode(decoder_input_tok_id, skip_special_tokens=False))
            for word_tok_ids, decoder_input_tok_id in zip(
                tok_output.token_ids, tok_output.decoder_token_ids)]
