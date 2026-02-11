
export OPENAI_API_KEY="YOUR_KEY_HERE"

python eval.py \
  --metadata "worldedit_bench/test.json" \
  --ori_root "worldedit_bench/images/" \
  --edit_root "output/" \
  --output "output/results.json" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --model "qwen-vl-max" \
  --workers 20 \
  --timeout 120 \
  --max_retries 3 \
  --resume
