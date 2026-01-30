export SERVER_WORLD_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $(($SERVER_WORLD_SIZE - 1)))

echo "SERVER_WORLD_SIZE: $SERVER_WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd script/qwen-image
bentoml serve --port 3000