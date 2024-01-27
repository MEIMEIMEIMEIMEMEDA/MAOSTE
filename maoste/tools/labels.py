from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

from maoste.data.data_types import LabelConvertorType
from maoste.data.data_types import LabelType

BIO_MAPPING = {
    'O': 0,
    'B-POS': 1,
    'I-POS': 2,
    'B-NEG': 1,
    'I-NEG': 2,
    'B-NEU': 1,
    'I-NEU': 2
}
REVERSE_BIO_MAPPING = ['O', 'B', 'I']

SENTIMENT_MAPPING = {
    'O': 0,
    'B-POS': 1,
    'I-POS': 1,
    'B-NEG': 2,
    'I-NEG': 2,
    'B-NEU': 3,
    'I-NEU': 3
}

REVERSE_SENTIMENT_MAPPING = ['O', 'POS', 'NEG', 'NEU']

BIO_SENTIMENT_MAPPING = {
    'O': 0,
    'B-POS': 1,
    'I-POS': 2,
    'B-NEG': 3,
    'I-NEG': 4,
    'B-NEU': 5,
    'I-NEU': 6
}
REVERSE_BIO_SENTIMENT_MAPPING = [
    'O', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU'
]

BIO_SENTIMENT_MAPPING = {
    'O': 0,
    'B-POS': 1,
    'I-POS': 2,
    'B-NEG': 3,
    'I-NEG': 4,
    'B-NEU': 5,
    'I-NEU': 6
}

BIO = [BIO_MAPPING, REVERSE_BIO_MAPPING]
SENTIMENT = [SENTIMENT_MAPPING, REVERSE_SENTIMENT_MAPPING]
BIO_SENTIMENT = [BIO_SENTIMENT_MAPPING, REVERSE_BIO_SENTIMENT_MAPPING]

# TODO
register = {
    LabelType.BIO: BIO,
    LabelType.SENTIMENT: SENTIMENT,
    LabelType.BIO_SENTIMENT: BIO_SENTIMENT,
}


def iob2(labels) -> List[Tuple[str, str]]:
  entities = []
  entity = None
  for label in labels:
    if label.startswith('B-'):
      entity = ('B', label[2:])
    elif label.startswith('I-'):
      entity = ('I', label[2:])
    else:
      entity = ('O', None)
    entities.append(entity)
  return entities


def bio_to_span_indice(labels) -> List[List[int]]:
  entities = []
  entity_type = None

  for i, (bio, entity_type) in enumerate(iob2(labels)):
    if entity_type is None:
      continue
    if bio == 'B':
      entities.append(
          [i, i + 1,
           REVERSE_SENTIMENT_MAPPING.index(entity_type.upper())])
    elif bio == 'I':
      entities[-1][1] = i + 1
  return entities


def words_bio_tag_to_span_text(words, labels) -> List[List[str]]:
  entities = []
  entity_type = None

  for i, (bio, entity_type) in enumerate(iob2(labels)):
    if entity_type is None:
      continue
    if bio == 'B':
      entities.append([i, i + 1, entity_type.upper()])
    elif bio == 'I':
      entities[-1][1] = i + 1
  return [
      [' '.join(words[st:ed]), entity_type] for st, ed, entity_type in entities
  ]


def get_label_list(labels):
  unique_labels = set()
  for label in labels:
    unique_labels = unique_labels | set(label)
  label_list = list(unique_labels)
  label_list.sort()
  return label_list


@dataclass
class LabelConvertorInputs:
  aspect: str
  label: str
  in_image: bool = None

  def tolist(self) -> List[Any]:
    return [self.aspect, self.label, self.in_image]


class LabelConvertor:
  in_image_sentence_template: str = None
  not_in_image_sentence_template: str = None

  @classmethod
  def encode(cls, inputs: LabelConvertorInputs) -> str:
    if inputs.in_image:
      return cls.in_image_sentence_template.format(aspect=inputs.aspect,
                                                   label=inputs.label)
    return cls.not_in_image_sentence_template.format(aspect=inputs.aspect,
                                                     label=inputs.label)

  @classmethod
  def decode(cls, sentence: str) -> LabelConvertorInputs:
    raise NotImplementedError('Decode Func')


class AspectLabelExistLC(LabelConvertor):
  in_image_sentence_template: str = '{aspect} is a {label} expression, which is in the image'
  not_in_image_sentence_template: str = '{aspect} is a {label} expression, which is not in the image'

  @classmethod
  def decode(self, sentence: str) -> LabelConvertorInputs:
    split_by_which = sentence.split('expression, which')
    if len(split_by_which) != 2:
      return LabelConvertorInputs('', '', None)
    split_aspect_label = split_by_which[0].split('is a')

    if len(split_aspect_label) != 2:
      return LabelConvertorInputs('', '', None)
    in_the_image = False if 'not in the image' in split_by_which[1] else True
    in_the_image = False if 'in the image' not in split_by_which[
        1] else in_the_image
    return LabelConvertorInputs(
        split_aspect_label[0].strip(),
        split_aspect_label[1].strip(),
        in_the_image,
    )


def get_label_convertor(convertor_type='classification') -> LabelConvertor:
  if isinstance(convertor_type, str):
    convertor_type = LabelConvertorType[convertor_type.upper()]
  if convertor_type == LabelConvertorType.ASPECT_LABEL_EXIST:
    return AspectLabelExistLC
  raise ValueError(f'Please check your input {list(LabelConvertorType)}')
