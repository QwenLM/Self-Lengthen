#!/bin/bash

max_iter=3
instruct_count=5000
base_model="meta-llama/Meta-Llama-3.1-8B-Instruct"

for i in "$@"
do
    case $i in
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

run_training() {
    local config_file=$1
    local iter=$2
    echo "Iter $iter: Start training $config_file"
    if [ $RANK -eq 0 ]; then
        FORCE_TORCHRUN=1 llamafactory-cli train $config_file
        echo "Iter $iter: Finish tokenizing $config_file"
    else
        python wait_all_finish.py --phase tokenize --config_file $config_file --iter $iter
    fi
    FORCE_TORCHRUN=1 NNODES=$WORLD_SIZE RANK=$RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
     llamafactory-cli train $config_file
    echo "Iter $iter: Finish training $config_file"
}

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32'
num_gpus=$(nvidia-smi --list-gpus | wc -l)
workers=$(( WORLD_SIZE * $num_gpus * 32 ))

for iter in $(seq 1 $max_iter)
do
    echo "Start iter $iter"

    bash run_server.sh --base_model=$base_model --iter=$((iter-1))

    if [ $RANK -eq 0 ]; then
        while true; do
            sleep 20s
            python llm_client.py
            if [ $? -eq 0 ]; then
                break
            else
                echo "Don't worry about the error above: vllm server has not started yet, wait 20s to retry"
            fi
        done
        echo "vllm server started"
        python step0_self_instruct.py --count $instruct_count --iter $iter --workers $workers
        python step1_gen_initial_responses.py --iter $iter --workers $workers
        python step2_ext_responses.py --iter $iter --workers $workers
    fi
    python wait_all_finish.py --iter $iter
    pkill -f -9 -e fastchat
    pkill -9 -e -f "from"

    if [ $RANK -eq 0 ]; then
        python mk_data.py --iter $iter
        python mk_train_config.py --iter $iter
    fi
    echo "Iter $iter: Finish making data"
    sleep 20s

    run_training ./train_config/gen_sft.yaml $iter
    run_training ./train_config/ext_sft.yaml $iter
done

bash run_collect_data.sh --max_iter=$max_iter --base_model=$base_model

pkill -f -9 -e fastchat
echo "All iter done"
rm controller.log openai_api_server.log model_worker_*.log
