#!/usr/bin/env bash

MODEL_PATH="../weights"  
TEST_JSON="worldedit_bench/test.json"
IMAGES_ROOT="worldedit_bench/images"     
OUTPUT_DIR="output/"  
LOG_JSON="output/log.json"

export CUDA_VISIBLE_DEVICES=0

python worldedit_infer.py \
  --model_path "${MODEL_PATH}" \
  --test_json "${TEST_JSON}" \
  --images_root "${IMAGES_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_json "${LOG_JSON}" \
  --cuda_visible_devices "0" \
  --max_memory "80GiB" \
  --seed 42 \
  --resume \
  --think \
  --max_think_token_n 1000 \
  --cfg_text_scale 5.0 \
  --cfg_img_scale 1.5 \
  --cfg_interval "0.0,1.0" \
  --timestep_shift 0.5 \
  --num_timesteps 50 \
  --cfg_renorm_min 0.0 \
  --cfg_renorm_type "text_channel"
