import torch

from maoste.core import CoreFactory
from maoste.core import CoreFactoryConfig
from maoste.core import get_mm_gen_args
from maoste.data.data_types import MetricType
from maoste.data.data_types import TaskType
from maoste.data.data_types import TrainerType
from maoste.model import VisionT5


def create_config(args, tokenizer):
  from transformers import T5Config

  config = T5Config.from_pretrained(args.lm_name)

  config.vinvl_region_num = args.vvl_region_num
  config.max_length = 200

  config.feat_dim = 2048
  config.img_anp_dim = 2089
  config.pos_dim = 36

  config.enable_global_img_feat = args.enable_global_img_feat

  config.dropout_rate = 0.1
  config.dropout = 0.1
  config.attention_dropout = 0.1
  config.activation_dropout = 0.1

  config.decoder_start_token_id = tokenizer.bos_token_id
  config.bos_token_id = tokenizer.bos_token_id
  config.eos_token_id = tokenizer.eos_token_id
  config.pad_token_id = tokenizer.pad_token_id
  config.ssep_token_id = tokenizer.ssep_token_id
  config.vsep_token_id = tokenizer.vsep_token_id

  config.not_in_image_token_ids = tokenizer.words2inputs(
      'not in the image')['input_ids']
  config.in_image_token_ids = tokenizer.words2inputs(
      'in the image')['input_ids']

  return config


def main(args):
  device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
  dataset_useful_fields = [
      'image_id',  # Image
      'cropped_images',
      'words',  # Text
      'bounding_boxes',  # vvl
      'box_features',
      'aspects',  # xml
      'boxes',
      'raw_target',
  ]
  if args.additional_fields:
    dataset_useful_fields.extend(args.additional_fields)
    dataset_useful_fields = list(set(dataset_useful_fields))
  model_config = CoreFactoryConfig(
      lm_name=args.lm_name,
      output_root_dir=args.output_root_dir,
      task_type=TaskType.MM_GEN,
      trainer_type=TrainerType.SEQ2SEQ,
      label_type=args.label_type,
      module_type=args.module_type,
      loss_type=MetricType.CLASSIFICATION,
      eval_type=MetricType.SPAN_TEXT_IMG,
      max_length=200,
      label_convertor_type=args.label_convertor_type,
      dataset_useful_fields=dataset_useful_fields,
  )
  model_factory = CoreFactory(model_config)
  model_factory.set_dataset_config(
      vvl_region_num=args.vvl_region_num,
      enable_global_img_feat=args.enable_global_img_feat)

  config = create_config(args, model_factory.tokenizer)
  model = VisionT5.from_pretrained(args.lm_name, config=config).to(device)
  model.resize_token_embeddings(len(model_factory.tokenizer))

  model_factory.run(skip_train=args.skip_train,
                    model=model,
                    args=model_factory.get_trainer_arguments(
                        batch_size=16, label_names=['labels', 'img_label']))


if __name__ == '__main__':
  main(get_mm_gen_args())
