#!/bin/bash

# Default parameters
OUTPUT_DIR="models/codeqwen25/sft-models"
MODEL_NAME="models/CodeLlama-7b-base-hf"
DATASET_NAME=data/rq3/codellama_merge.jsonl


CONFIG_FILE="examples/accelerate_configs/deepspeed_zero3.yaml"
MAX_STEPS=-1
BATCH_SIZE=1
SEQ_LEN=16384


# Set your number of GPUs per node and servers here
# SERVERS="10.103.255.3:0 10.103.255.4:1 10.103.255.5:2 10.103.255.6:3 10.103.255.7:4" # Add all server IPs and their node ranks
SERVERS="10.103.255.2:0"
IMAGE_NAME="trl_env:0910"
WORKSPACE_PATH=""
WORK_DIR="aiXcoder-colt"
CONTAINER_NAME="trl-sft"
WANDB_API_KEY=""

# Parse servers and set master address
IFS=' ' read -r -a ADDR_ARRAY <<< "$SERVERS"
FIRST_SERVER="${ADDR_ARRAY[0]}"
IFS=':' read -r -a SERVER_IP_RANK <<< "$FIRST_SERVER"
MASTER_ADDR="${SERVER_IP_RANK[0]}"
NNODES=${#ADDR_ARRAY[@]}

# Verify master address
ifconfig ib1 | grep "inet " | awk '{print $2}' | cut -d ':' -f 2 | grep -q $MASTER_ADDR
if [[ $? -ne 0 ]]; then
  echo "Master Address is Wrong"
  exit 1
fi

echo "Master Address is Right"
rm -f core.*
set -x

# Generate random master port
MASTER_PORT="160$((RANDOM % 100))"

echo "Starting multi-node training..."

# Loop through each server and start the process
for SERVER in $SERVERS
do
    ADDR=${SERVER%:*}
    NODE_RANK=${SERVER#*:}
    IB="ib1" # Update IB interface if needed

    # Environment variables for NCCL and other settings
    # ENV_ARGS="CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_SOCKET_IFNAME=$IB GLOO_SOCKET_IFNAME=$IB NCCL_DEBUG=INFO"
    ENV_ARGS=""

    # Command for running accelerate with appropriate settings
    CMD="""accelerate launch --config_file $CONFIG_FILE \
    --num_processes $((8 * NNODES)) --num_machines $NNODES --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --mixed_precision bf16 \
    /nfs100/zhuhao/trl_debug/trl-main/trl-main/examples/scripts/0218/sft_codellama.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 8 \
    --max_seq_length $SEQ_LEN \
    --num_train_epochs 2 \
    --attn_implementation flash_attention_2 \
    --logging_steps 5 \
    --save_only_model \
    --save_steps 360 \
    --learning_rate 2e-6 \
    --use_liger_kernel \
    --gradient_checkpointing \
    --bf16 \
    --weight_decay 0.01 \
    --report_to wandb \
    --max_grad_norm 1.0
"""
    # --bf16_full_eval \
    # --eval_steps 50 \
    # --evaluation_strategy steps \
    # --per_device_eval_batch_size 1 \

    # Extracting the last part of the server IP for unique naming
    OLD_IFS="$IFS"
    IFS='.'
    read -ra SERVER_KEY <<< "$ADDR"
    SERVER_KEY="${SERVER_KEY[3]}"
    IFS="$OLD_IFS"

    # Combining environment variables with the command
    # RUN_CMD="$ENV_ARGS TOKENIZERS_PARALLELISM=true $CMD"
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


    # Define experiment name and log file
    EXP_NAME=sft_0218
    LOG_FILE="$WORK_DIR/logs/codellama_$EXP_NAME.log"

    if [[ $ADDR != $MASTER_ADDR ]]; then
        # For non-master nodes, run the command via SSH and Docker
        ssh -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null $ADDR "
        docker kill $CONTAINER_NAME-$SERVER_KEY || true
        docker rm $CONTAINER_NAME-$SERVER_KEY || true
        sleep 4
        docker run -d -it --gpus \"device=0,1,2,3,4,5,6,7\" --net=host --name $CONTAINER_NAME-$SERVER_KEY \
            -v /dev/shm:/dev/shm \
            -v $WORKSPACE_PATH:$WORKSPACE_PATH \
            -v $DATASET_NAME:$DATASET_NAME \
            -v $WORK_DIR:$WORK_DIR \
            -v $MODEL_NAME:$MODEL_NAME \
            -e WANDB_API_KEY=$WANDB_API_KEY \
            -w $WORK_DIR $IMAGE_NAME /bin/bash -c '$RUN_CMD' 
        docker logs -f $CONTAINER_NAME-$SERVER_KEY >& $LOG_FILE &  
        " &
    else
        # For the master node, run directly
        docker kill $CONTAINER_NAME-$SERVER_KEY || true
        docker rm $CONTAINER_NAME-$SERVER_KEY || true
        sleep 4
        docker run -d -it --gpus \"device=0,1,2,3,4,5,6,7\" --net=host --name $CONTAINER_NAME-$SERVER_KEY \
            -v /dev/shm:/dev/shm \
            -v $WORKSPACE_PATH:$WORKSPACE_PATH \
            -v $WORK_DIR:$WORK_DIR \
            -v $DATASET_NAME:$DATASET_NAME \
            -v $MODEL_NAME:$MODEL_NAME \
            -e WANDB_API_KEY=$WANDB_API_KEY \
            -w $WORK_DIR $IMAGE_NAME /bin/bash -c "$RUN_CMD" 
        docker logs -f $CONTAINER_NAME-$SERVER_KEY >& $LOG_FILE &  
    fi
done




