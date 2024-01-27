import argparse
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from enum import Enum
from functools import partial
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import PreTrainedModel
from transformers import Seq2SeqTrainingArguments
from transformers import set_seed
from transformers import Trainer
from transformers import TrainingArguments

from maoste.data.data_types import LabelConvertorType
from maoste.data.data_types import LabelType
from maoste.data.data_types import MetricType
from maoste.data.data_types import ModuleType
from maoste.data.data_types import TaskType
from maoste.data.data_types import TrainerType
from maoste.dataset.twitter import TwitterDataset
from maoste.dataset.twitter import TwitterDatasetConifg
from maoste.tools.labels import register as label_register
from maoste.tools.tokenizer import TokenizerConfig
from maoste.tools.tokenizer import ToknizerWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class CoreFactoryConfig:
  # Language model name
  lm_name: str = 'bert-base-uncased'
  # Output root directory
  output_root_dir: str = './log'
  monitor_metric: str = 'f1'

  # Enum control return
  task_type: TaskType = TaskType.TEXT_TOKEN_CLS
  trainer_type: TrainerType = TrainerType.COMMON
  module_type: ModuleType = ModuleType.NONE
  label_type: LabelType = LabelType.BIO
  label_convertor_type: LabelConvertorType = LabelConvertorType.ASPECT_LABEL_EXIST
  loss_type: MetricType = MetricType.CLASSIFICATION
  eval_type: MetricType = MetricType.CLASSIFICATION

  # Dataset root directory
  train_root: str = 'dataset/npz/train'
  val_root: str = 'dataset/npz/dev'
  test_root: str = 'dataset/npz/test'
  # Dataset useful fields
  dataset_useful_fields: List[str] = field(
      default_factory=lambda: ['image_id', 'words', 'word_ids', 'raw_target'])
  max_length: int = 80

  # Label config
  num_labels: int = 3
  label_indice_mapping: Dict[str, int] = field(
      default_factory=lambda: label_register[LabelType.BIO][0])
  reverse_labels_mapping: List[int] = field(
      default_factory=lambda: label_register[LabelType.BIO][1])
  label_name_mapping: Dict[str, str] = field(default_factory=lambda: {
      'POS': 'positive',
      'NEG': 'negative',
      'NEU': 'neutral'
  })

  @property
  def output_dir(self):
    return os.path.join(
        self.output_root_dir,
        self.task_type.name,
        self.module_type.name,
        self.lm_name.replace('/', '_'),
        self.label_type.name,
        self.monitor_metric,
    )

  @property
  def model_ckpts(self):
    model_ckpts = [
        exp_dir / 'pytorch_model.bin'
        for exp_dir in Path(self.output_dir).glob('*')
    ]
    return list(filter(lambda x: x.exists(), model_ckpts))

  def _set_label_config(self):
    self.label_indice_mapping, self.reverse_labels_mapping = label_register[
        self.label_type].copy()
    num_labels = len(set(self.label_indice_mapping.values()))
    if ModuleType.CRF.name in self.module_type.name:
      # add padding in -1 position
      self.reverse_labels_mapping.append('PAD')
      num_labels += 1
    self.num_labels = num_labels

  @classmethod
  def from_yaml(cls, fn: str):
    return cls(**OmegaConf.load(fn))

  def to_dict(self):
    return asdict(self)

  def save(self):
    os.makedirs(self.output_dir, exist_ok=True)
    conf = OmegaConf.create(self.to_dict())
    OmegaConf.save(config=conf,
                   f=os.path.join(self.output_dir, 'core_factory_config.yaml'))

  @classmethod
  def build_config_by_args(cls, **kwargs):
    return

  def __post_init__(self):
    for f in fields(self):
      v = getattr(self, f.name)
      if isinstance(f.default, Enum) and isinstance(v, str):
        setattr(self, f.name, f.type[v.upper()])
    self._set_label_config()
    self.save()



class CoreFactory:
  """TODO refactor by task type."""

  def __init__(self, config: CoreFactoryConfig) -> None:
    self.config = config
    self._trainer_kwargs = None
    self._tokenizer = None
    self._dataset_config = TwitterDatasetConifg(
        self.config.train_root,
        self.tokenizer,
        fields=self.config.dataset_useful_fields,
    )

  @property
  def device(self):
    return torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

  @property
  def tokenizer(self):
    if self._tokenizer is None:
      self._tokenizer = self.get_tokenizer(self.config.max_length)
    return self._tokenizer

  @property
  def dataset_cls(self) -> TwitterDataset:
    return TwitterDataset

  @property
  def trainer_cls(self):
    if self.config.task_type == TaskType.MM_GEN:
      from maoste.tools.trainer import Seq2SeqWithVisionTrainer
      return Seq2SeqWithVisionTrainer
    raise RuntimeError(f'Please check your input {list(TaskType)}')

  def get_tokenizer(self, max_length=80) -> ToknizerWrapper:
    tokenizer = AutoTokenizer.from_pretrained(self.config.lm_name,
                                              add_prefix_space=True,
                                              use_fast=True)
    tokenizer_cfg = TokenizerConfig(
        self.config.label_indice_mapping,
        self.config.reverse_labels_mapping,
        self.config.label_convertor_type,
        self.config.label_name_mapping,
        max_length,
    )
    tokenizer_wrapper = ToknizerWrapper(
        tokenizer,
        tokenizer_cfg,
    )
    return tokenizer_wrapper

  def get_model(self, **kwargs) -> PreTrainedModel:
    task_cls = None
    if self.config.task_type == TaskType.TEXT_TOKEN_CLS:
      task_cls = AutoModelForTokenClassification
    elif self.config.task_type == TaskType.TEXT_SEQ2SEQ:
      task_cls = AutoModelForSeq2SeqLM
    elif self.config.task_type == TaskType.MM_TWO_STAGE:
      task_cls = AutoModel
      return task_cls.from_pretrained(self.config.align_model_name,
                                      ignore_mismatched_sizes=True,
                                      **kwargs)
    model = task_cls.from_pretrained(self.config.lm_name,
                                     ignore_mismatched_sizes=True,
                                     **kwargs)
    model.resize_token_embeddings(len(self.tokenizer))
    return model

  def get_latest_pretrained_ckpt(self) -> PreTrainedModel:
    # TODO(wolf): best
    ckpt_dirs = sorted(Path(self.config.output_dir).glob('checkpoint*'))
    _loaded = False
    for ckpt_dir in ckpt_dirs[::-1]:
      ckpt = ckpt_dir / 'pytorch_model.bin'
      if ckpt.exists():
        logger.info(f'Load model weights from {ckpt}')
        return torch.load(str(ckpt))
    if not _loaded:
      logger.warning('Missing pretrained "pytorch_model.bin" weight')

  def get_eval_metric(self, **kwargs) -> callable:
    from maoste.tools.eval import get_eval
    return partial(get_eval(self.config.eval_type), **kwargs)

  def get_data_collator(self, **kwargs) -> callable:
    from maoste.tools.collator import get_collator
    return partial(get_collator(self.config.task_type), **kwargs)

  def set_dataset_config(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self._dataset_config, k, v)

  def get_train_dataset(self) -> TwitterDataset:
    self._dataset_config.root = self.config.train_root
    return self.dataset_cls(self._dataset_config)

  def get_val_dataset(self) -> TwitterDataset:
    self._dataset_config.root = self.config.val_root
    return self.dataset_cls(self._dataset_config)

  def get_test_dataset(self) -> TwitterDataset:
    self._dataset_config.root = self.config.test_root
    return self.dataset_cls(self._dataset_config)

  def get_trainer_arguments(
      self,
      num_workers: int = 16,
      batch_size: int = None,
      **kwargs,
  ) -> Union[TrainingArguments, Seq2SeqTrainingArguments]:
    trainer_args_cls = None
    # Trainer arguments
    if not isinstance(batch_size, int):
      batch_size = num_workers * 2
    trainer_args = dict(
        output_dir=self.config.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=35,
        evaluation_strategy='steps',
        metric_for_best_model=self.config.monitor_metric,
        seed=2023,
        save_total_limit=2,
        save_steps=num_workers * 16,
        eval_steps=num_workers * 8,
        load_best_model_at_end=True,
        dataloader_num_workers=num_workers,
        label_names=['labels'],
    )
    if self.config.trainer_type == TrainerType.COMMON:
      trainer_args_cls = TrainingArguments
    elif self.config.trainer_type == TrainerType.SEQ2SEQ:
      trainer_args_cls = Seq2SeqTrainingArguments
      trainer_args.update(
          dict(
              predict_with_generate=True,
              generation_num_beams=1,  # greedy
          ))
    if trainer_args_cls is None:
      raise TypeError('Please input valid TrainerType')
    trainer_args.update(kwargs)
    return trainer_args_cls(**trainer_args)

  def init_trainer_default_kwargs(self):
    trainer_kwargs = dict(
        args=self.get_trainer_arguments(),
        train_dataset=self.get_train_dataset(),
        eval_dataset=self.get_val_dataset(),
        data_collator=self.get_data_collator(),
        compute_metrics=self.get_eval_metric(tokenizer=self.tokenizer,
                                             ignore_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    return trainer_kwargs

  def get_trainer_kwargs(self, **kwargs):
    if self._trainer_kwargs is None:
      self._trainer_kwargs = self.init_trainer_default_kwargs()

    self._trainer_kwargs.update(kwargs)
    return self._trainer_kwargs

  def get_trainer(self, **kwargs):
    return self.trainer_cls(**self.get_trainer_kwargs(**kwargs))

  def run(self, skip_train: bool = False, **kwargs):
    trainer = self.get_trainer(**kwargs)
    if not skip_train:
      trainer.train()
    self.evaluate_with_trainer(trainer)

  def evaluate_with_trainer(self, trainer: Trainer):
    import json

    logger.info('=' * 10)
    logger.info('Start Evaluation')
    logger.info('=' * 10)
    pbar = tqdm(self.config.model_ckpts[:1], desc='Evaluate in test dataset')
    for ckpt in pbar:
      eval_res_file = ckpt.parent / 'eval_test.json'

      trainer.model.load_state_dict(torch.load(str(ckpt)))
      with open(eval_res_file, 'w', encoding='utf-8') as fp:
        json.dump(trainer.evaluate(self.get_test_dataset()), fp)


def default_arg_parser() -> argparse.ArgumentParser:
  set_seed(2023)
  parser = argparse.ArgumentParser(description='Process some arguments.')
  parser.add_argument('--output-root-dir', type=str, default='./log')
  parser.add_argument('--lm-name',
                      type=str,
                      default='t5-base',
                      help='Model name')
  parser.add_argument('--label-type',
                      type=str,
                      default=LabelType.BIO_SENTIMENT.name,
                      choices=[mt.name for mt in LabelType],
                      help='Label name')
  parser.add_argument('--module-type',
                      default=ModuleType.NONE.name,
                      choices=[mt.name for mt in ModuleType],
                      help='Sub modules')
  parser.add_argument('--skip-train',
                      action='store_true',
                      help='Enable training')
  return parser


def get_mm_gen_args() -> argparse.Namespace:
  parser = default_arg_parser()
  parser.add_argument('--vvl-region-num', type=int, default=31)
  parser.add_argument('--anp-num', type=int, default=5)
  parser.add_argument('--additional-fields', nargs='*', type=str, default=["anp", "image_caption"])
  parser.add_argument('--label-convertor-type',
                      default=LabelConvertorType.ASPECT_LABEL_EXIST.name,
                      choices=[item.name for item in LabelConvertorType],
                      type=str)
  parser.add_argument('--enable-global-img-feat', action='store_true')
  return parser.parse_args()

