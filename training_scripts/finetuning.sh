#!/bin/bash
# images: python 3.10 torch 2.1 cuda 12.1 deepspeed 0.10.0

set -x
. ~/.bashrc
wandb disabled

export NCCL_DEBUG="INFO"

world_size=`expr $NODE_NUM \* $GPU_NUM_PER_NODE`
echo "NODE_NUM: "$NODE_NUM
echo "GPU_NUM_PER_NODE: "$GPU_NUM_PER_NODE
echo "world_size: "$world_size
echo "CHIEF_IP: "$CHIEF_IP
echo "GPU INDEX: "$INDEX

pip install -r requirements.txt

# todo: Set your llms root dir
llm_root=/your/path/to/llms_root_dir/
model_name=CodeLlama-7b-hf

BASE_MODEL_DIR=$llm_root/$model_name
DATA_DIR=./data
OUTPUT_DIR=./output

echo $PWD

DATA_PATH=$DATA_DIR/processed/sql-llama-instruct-v0.5.jsonl
LLAMA_MODEL_DIR=$BASE_MODEL_DIR

GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count());")
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=${MASTER_PORT:-6100}
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MAX_STEPS=1000

DEEPSPEED_CONFIG="configs/default_offload_opt_param.json"
BATCH_SIZE=32 # 这个只是用来计算 GRAD_ACCU，每次参数更新所用的总数据量即为 Batch_Size大小
MICRO_BATCH_SIZE=1 # 这个才是每张卡实际得到的样本量
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE)) # 按照8卡 bs32计算，GRAD_ACCU=4

LR=2e-5
WARMUP_RATIO=0.03
WEIGHT_DECAY=0.0
MAX_LENGTH=4300 # 这个参数和显存占用有直接关系
CKPT_OUTPUT_DIR="$OUTPUT_DIR/macsql-lr${LR}-wr${WARMUP_RATIO}-wd${WEIGHT_DECAY}-bsz${BATCH_SIZE}-maxlen${MAX_LENGTH}/"
LOG_OUTPUT_DIR="$OUTPUT_DIR/logs/"

echo $CKPT_OUTPUT_DIR
echo "WORLD_SIZE" $WORLD_SIZE "MICRO BATCH SIZE" $MICRO_BATCH_SIZE "GRAD_ACCU" $GRAD_ACCU
echo $DISTRIBUTED_ARGS

torchrun $DISTRIBUTED_ARGS finetuning.py \
    --model_name_or_path  $BASE_MODEL_DIR \
    --data_path $DATA_PATH \
    --model_max_length $MAX_LENGTH \
    --output_dir $CKPT_OUTPUT_DIR \
    --num_train_epochs 10 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCU} \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_dir $LOG_OUTPUT_DIR \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed $DEEPSPEED_CONFIG \
    --bf16 True \
    --tf32 True > ./log.txt

echo "Training done!"