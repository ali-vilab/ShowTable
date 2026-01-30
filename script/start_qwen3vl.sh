CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/wulanchabu/liuzh/models/Qwen3-VL-235B-A22B-Instruct \
  --trust-remote-code \
  --seed 42 \
  --max_model_len 16384 \
  --served-model-name Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --port 23333 \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image": 4}'
