# producer
mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=127.0.0.1 \
    --rpc_port=35551 \
    --http_metadata_server_port=35880 \
    --metrics_port=35903


export FLASHINFER_DISABLE_VERSION_CHECK=1
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=68719476736
export MOONCAKE_LOCAL_BUFFER_SIZE=1073741824

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES=2 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-27B \
    --dtype bfloat16 \
    --trust-remote-code \
    --tp-size 1 \
    --context-length 16384 \
    --attention-backend flashinfer \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --disable-cuda-graph \
    --enable-spec-capture \
    --spec-capture-method dflash \
    --spec-capture-aux-layer-ids 1 16 31 46 61 \
    --host 127.0.0.1

# Check sglang capture whether run success.
curl -f http://127.0.0.1:30000/health


RUN_ID=qwen3.6-27b-domino-split-disaggregated-08082328
OUTPUT_DIR=outputs/${RUN_ID}
CONTROL_DIR=${OUTPUT_DIR}/control
CONSUMER_STATE_DIR=${OUTPUT_DIR}/consumer-state

# producer
CUDA_VISIBLE_DEVICES=2 specforge train \
    -c examples/configs/qwen3.6-27b-domino-online.yaml \
    --role producer \
    "run_id=${RUN_ID}" \
    "output_dir=${OUTPUT_DIR}" \
    "deployment.disaggregated.control_dir=${CONTROL_DIR}" \
    "deployment.disaggregated.consumer_state_dir=${CONSUMER_STATE_DIR}"


# consumer
CUDA_VISIBLE_DEVICES=3 specforge train \
    -c examples/configs/qwen3.6-27b-domino-online.yaml \
    --role consumer \
    "run_id=${RUN_ID}" \
    "output_dir=${OUTPUT_DIR}" \
    "deployment.disaggregated.control_dir=${CONTROL_DIR}" \
    "deployment.disaggregated.consumer_state_dir=${CONSUMER_STATE_DIR}"
