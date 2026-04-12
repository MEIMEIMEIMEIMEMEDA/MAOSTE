#!/bin/bash
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=true
script_dir=$(dirname "$0")
anp_nums=(5)
vvl_region_nums=(31)
additional_fields=("anp image_caption")
label_convertor_type=("ASPECT_EXIST_LABEL")

counter=0

for anp_num in "${anp_nums[@]}"; do
  for vvl_region_num in "${vvl_region_nums[@]}"; do
    for additional_field in "${additional_fields[@]}"; do
      for label_convertor in "${label_convertor_type[@]}"; do

        python $script_dir/main.py --output-root-dir "e$counter" --lm-name t5-base --anp-num $anp_num --vvl-region-num $vvl_region_num --additional-fields $additional_field --label-convertor-type $label_convertor --enable-global-img-feat
        ((counter++))

      done
    done
  done
done
