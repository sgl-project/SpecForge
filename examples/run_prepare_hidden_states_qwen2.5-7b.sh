SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

cd $ROOT_DIR
export PYTHONPATH=$ROOT_DIR

MODEL_PATH=Qwen2.5-7B-Instruct
MODEL_NAME=Qwen2.5-7B-Instruct
DATASET_PATH=/datasets/all.jsonl
OUTPUT_ROOT_DIR=/specforge_output/$MODEL_NAME/all
NUM_SAMPLES=20000
TP_SIZE=1
MAX_LENGTH=16384
DRAFT_CONFIG_PATH=./configs/qwen2.5-7b-eagle3.json

OUTPUT_DIR=$OUTPUT_ROOT_DIR/outputs

NPROC_PER_NODE=2

torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    scripts/prepare_hidden_states_hf.py \
    --model-path $MODEL_PATH \
    --enable-aux-hidden-states \
    --data-path $DATASET_PATH \
    --chat-template qwen \
    --max-length $MAX_LENGTH \
    --tp-size $TP_SIZE \
    --batch-size 1 \
    --mem-frac=0.5 \
    --output-path $OUTPUT_ROOT_DIR/cache/hidden_states \
    --num-samples $NUM_SAMPLES \
