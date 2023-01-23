#!/bin/bash

# Set these in the launch script
ROOT_DIR=$ROOT_DIR
NUM_COLLECT_JOBS=$NUM_CT_COLLECT_JOBS
GLOBAL_SEED=$GLOBAL_SEED

OUTPUT_DIR="${ROOT_DIR}"/output
mkdir -p "$OUTPUT_DIR"

export REVERB_PORT=8008
export REVERB_SERVER="127.0.0.1:${REVERB_PORT}"
export NETLIST_FILE=./circuit_training/environment/test_data/ariane/netlist.pb.txt
export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane/initial.plc

# Use local tensorboard configuration instead of tensorboard dev
#tmux send-keys -t "tb_job" "tensorboard dev upload --logdir ./logs" Enter
#tmux send-keys -t "tb_job" "yes" Enter

# Ensure that the reverb server starts without using GPU
CUDA_VISIBLE_DEVICES=-1 python3 -m circuit_training.learning.ppo_reverb_server \
  --root_dir="${ROOT_DIR}" \
  --global_seed="${GLOBAL_SEED}" \
  --port=${REVERB_PORT} &>"${OUTPUT_DIR}/reverb_server_output" &

# Start all of the jobs in the background of the current shell
python3 -m circuit_training.learning.train_ppo \
  --root_dir="${ROOT_DIR}" \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --num_episodes_per_iteration=16 \
  --global_batch_size=64 \
  --netlist_file=${NETLIST_FILE} \
  --global_seed="${GLOBAL_SEED}" \
  --init_placement=${INIT_PLACEMENT} &>"${OUTPUT_DIR}/train_job_output" &

# iterate from 0 to $NUM_COLLECT_JOBS
for i in $(seq 0 $(($NUM_COLLECT_JOBS - 1))); do
  printf -v padded "%02d" $i
  CUDA_VISIBLE_DEVICES=-1 python3 -m circuit_training.learning.ppo_collect \
    --root_dir="${ROOT_DIR}" \
    --replay_buffer_server_address=${REVERB_SERVER} \
    --variable_container_server_address=${REVERB_SERVER} \
    --task_id="${i}" \
    --netlist_file=${NETLIST_FILE} \
    --global_seed="${GLOBAL_SEED}" \
    --init_placement=${INIT_PLACEMENT} &>"${OUTPUT_DIR}/collect_job_${padded}" &
done

# Start final job in the foreground of the current shell
CUDA_VISIBLE_DEVICES=-1 python3 -m circuit_training.learning.eval \
  --root_dir="${ROOT_DIR}" \
  --variable_container_server_address=${REVERB_SERVER} \
  --netlist_file=${NETLIST_FILE} \
  --global_seed="${GLOBAL_SEED}" \
  --init_placement=${INIT_PLACEMENT} &>"${OUTPUT_DIR}/eval_job"
