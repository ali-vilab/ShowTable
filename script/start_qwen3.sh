CUDA_VISIBLE_DEVICES=0,1 \
python -m vllm.entrypoints.openai.api_server \
    --model ckpt/qwen3-8b-trained  \
    --trust-remote-code \
    --max_model_len 8192 \
    --served-model-name qwen3 \
    --tensor-parallel-size 2 \
    --port 23333 \
    --gpu-memory-utilization 0.85 

