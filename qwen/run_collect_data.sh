#!/bin/bash

total_count=5000
instruct_count=500
base_model="Qwen/Qwen2-7B-Instruct"
max_iter=3

sleep 10s

for i in "$@"
do
    case $i in
        --total_count=*)
        total_count="${i#*=}"
        shift
        ;;
        --max_iter=*)
        max_iter="${i#*=}"
        shift
        ;;
        --instruct_count=*)
        instruct_count="${i#*=}"
        shift
        ;;
        --base_model=*)
        base_model="${i#*=}"
        shift
        ;;
        *)
        echo "Unknown option $i"
        exit 1
        ;;
    esac
done


bash run_server.sh --base_model=$base_model --iter=$max_iter

while true; do
    sleep 20s
    python llm_client.py
    if [ $? -eq 0 ]; then
        break
    else
        echo "vllm server has not started, wait 20s to retry"
    fi
done

num_gpus=$(nvidia-smi --list-gpus | wc -l)
workers=$(( WORLD_SIZE * $num_gpus * 32 ))

count=0
while [ $count -lt $total_count ]
do
    echo "vllm server started"
    iter=$(find results -mindepth 1 -maxdepth 1 -type d | wc -l)
    iter=$((iter+10000))
    mkdir -p results/iter_$iter
    python step0_self_instruct.py --count $instruct_count --iter $iter --workers $workers
    python step1_gen_initial_responses.py --iter $iter --workers $workers
    count=$(( $count + $instruct_count ))
done
