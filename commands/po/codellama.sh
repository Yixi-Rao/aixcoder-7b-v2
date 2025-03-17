#!/bin/bash
# This script runs an SFT example end-to-end on a single machine with 8 GPUs



OUTPUT_DIR="models/align_models/codellama/0307_dpo"
MODEL_NAME=models/codeqwen25/sft-models/checkpoint-1780
DATASET_NAME="dpo/Reject_Sample/codellama-0223-python/hf_dataset_multi_lang"



CONFIG_FILE="examples/accelerate_configs/deepspeed_zero3.yaml"
WORKSPACE_PATH=""
WORK_DIR="aiXcoder-colt"
IMAGE_NAME="trl_env:0910"
CONTAINER_NAME="trl-dpo-lora-training"
WANDB_API_KEY=""



MAX_STEPS=-1
BATCH_SIZE=1
SEQ_LEN=12384

# Define log file
EXP_NAME="codellama_0307_dpo"
LOG_FILE="$WORK_DIR/logs/${CONTAINER_NAME}_${EXP_NAME}.log"

set -x

# NCCL environment variables
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TIMEOUT=3600

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True





# Generate a random master port for local communication
MASTER_PORT="160$((RANDOM % 100))"

# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_SOCKET_IFNAME=ib1 GLOO_SOCKET_IFNAME=ib1

# Command to launch training
CMD="""NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_TIMEOUT=3600 NCCL_DEBUG=INFO  accelerate launch --config_file $CONFIG_FILE \
--num_processes 8 --num_machines 1 --machine_rank 0 \
--main_process_port $MASTER_PORT --mixed_precision bf16 \
$WORK_DIR/examples/scripts/dpo_lora_all_module.py \
--model_name $MODEL_NAME \
--dataset_name $DATASET_NAME \
--output_dir $OUTPUT_DIR \
--max_steps $MAX_STEPS \
--per_device_train_batch_size $BATCH_SIZE \
--gradient_accumulation_steps 16 \
--max_prompt_length 12384 \
--max_length 12384 \
--num_train_epochs 1 \
--attn_implementation flash_attention_2 \
--save_strategy epoch \
--learning_rate 1e-4 \
--bf16 \
--bf16_full_eval \
--logging_steps 3 \
--log_level info \
--save_only_model \
--gradient_checkpointing \
--no_remove_unused_columns \
--torch_dtype float16 \
--use_liger_kernel \
--use_peft \
--lora_r 32 \
--lora_alpha 16 \
--rpo_alpha 1 \
--warmup_steps 520 \
--do_eval True \
--eval_steps 200 \
--evaluation_strategy steps \
--per_device_eval_batch_size 1 \
--beta 0.9
"""

RUN_CMD="
    export HTTP_PROXY=http://10.103.255.1:7890 &&
    export HTTPS_PROXY=http://10.103.255.1:7890 &&
    export http_proxy=http://10.103.255.1:7890 &&
    export https_proxy=http://10.103.255.1:7890 &&
    echo \"==== Upgrading pip ====\" &&
    python -m pip install --upgrade pip &&
    echo \"==== Installing wandb ====\" &&
    pip install wandb==0.15.0 &&
    echo \"==== Starting Training ====\" &&
    $CMD
"



# Docker command to run the training
docker kill $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true
sleep 4
docker run -d -it --gpus all --net=host --name $CONTAINER_NAME \
    -v /dev/shm:/dev/shm \
    -v $WORKSPACE_PATH:$WORKSPACE_PATH \
    -v $OUTPUT_DIR:$OUTPUT_DIR \
    -v $DATASET_NAME:$DATASET_NAME \
    -v $WORK_DIR:$WORK_DIR \
    -v $MODEL_NAME:$MODEL_NAME \
    -e TOKENIZERS_PARALLELISM=true \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -w $WORK_DIR $IMAGE_NAME /bin/bash -c "$RUN_CMD"

# Follow logs
docker logs -f $CONTAINER_NAME >& $LOG_FILE &
