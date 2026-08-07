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

CUDA_VISIBLE_DEVICES=0 \
  python -m sglang.launch_server \
    --model-path /disk3/wjp/pretrained_models/Qwen3.6-27B \
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

# 检查是否sglang capture是否成功启动
curl -f http://127.0.0.1:30000/health

CUDA_VISIBLE_DEVICES=0 specforge train \
    -c examples/configs/qwen3.6-27b-domino-split-disaggregated.yaml \
    --role producer

# consumer
CUDA_VISIBLE_DEVICES=1 specforge train \
    -c examples/configs/qwen3.6-27b-domino-split-disaggregated.yaml \
    --role consumer


# 0721更新：1. 更新了_result中item操作的时机，不用每次都进行同步 2. 增大了runtime的batch为2