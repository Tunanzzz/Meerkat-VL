
MODEL_PATH="xxxxxx"

GPU_IDS=(1)
BASE_PORT=xxx

LOG_DIR="./logs_qwen_vl"
mkdir -p $LOG_DIR

MAX_MODEL_LEN=8192
GPU_UTIL=0.6


echo "Starting deployment for model: $MODEL_PATH"
echo "Target GPUs: ${GPU_IDS[@]}"
echo "Logs will be saved to: $LOG_DIR"
echo "----------------------------------------"

for i in "${!GPU_IDS[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    CURRENT_PORT=$((BASE_PORT + i))
    LOG_FILE="$LOG_DIR/server_gpu_${GPU_ID}_port_${CURRENT_PORT}.log"

    echo "Launching server on GPU $GPU_ID | Port: $CURRENT_PORT..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID \
    vllm serve $MODEL_PATH \
        --host 0.0.0.0 \
        --port $CURRENT_PORT \
        --tensor-parallel-size 1 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --served-model-name "xxxx" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$LOG_DIR/server_gpu_${GPU_ID}.pid"
done


echo "----------------------------------------"
echo "All servers are launching in the background."
echo "Please check log files in $LOG_DIR for status."
echo "Use 'tail -f $LOG_DIR/*.log' to monitor startup."
