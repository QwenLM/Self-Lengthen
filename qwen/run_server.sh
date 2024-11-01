#!/bin/bash

base_model='Qwen/Qwen2-7B-Instruct'
iter=0

for i in "$@"
do
    case $i in
        --base_model=*)
        base_model="${i#*=}"
        shift
        ;;
        --iter=*)
        iter="${i#*=}"
        shift
        ;;
        *)
        echo "Unknown option $i"
        exit 1
        ;;
    esac
done

num_gpus=$(nvidia-smi --list-gpus | wc -l)

pkill -f -9 -e fastchat
pkill -f -9 -e from
my_ip=$(python -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')

master_addr=${MASTER_ADDR:-localhost}
if [[ "${master_addr}" == "localhost" ]]; then
    master_ip='127.0.0.1'
else
    master_ip=$(getent hosts ${master_addr} | awk '{print $1}')
fi

controller_port=21001
controller_address="http://${master_ip}:${controller_port}"

echo "my_ip: $my_ip"
echo "controller_address: $controller_address"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FASTCHAT_WORKER_API_TIMEOUT=9999

echo "RANK: $RANK"
if [ $RANK -eq 0 ]; then
    python -m fastchat.serve.controller \
        --host 0.0.0.0 \
        --port $controller_port > /dev/null 2>&1 &
    python -m fastchat.serve.openai_api_server \
        --controller-address $controller_address \
        --host 0.0.0.0 \
        --port 8000 > /dev/null 2>&1 &
fi

port=8080
model_size=$(echo "$base_model" | sed -n 's/.*-\([0-9]\+\)B.*/\1/p')
echo "model_size: $model_size"

if [ $iter -eq 0 ]; then
    if [ $model_size -gt 10 ]; then
        python -m fastchat.serve.vllm_worker \
            --model-names $base_model \
            --model-path $base_model \
            --conv-template qwen-7b-chat \
            --tensor-parallel-size $num_gpus \
            --controller-address $controller_address \
            --worker-address http://$my_ip:$port \
            --host $my_ip \
            --port $port > /dev/null 2>&1 &
    else
        for i in $(seq 0 $(($num_gpus - 1)))
        do
            echo "GPU $i is going to start"
            port=$((8080+$i))
            CUDA_VISIBLE_DEVICES=$i python -m fastchat.serve.vllm_worker \
                --model-names $base_model \
                --model-path $base_model \
                --conv-template qwen-7b-chat \
                --controller-address $controller_address \
                --worker-address http://$my_ip:$port \
                --host $my_ip \
                --port $port > /dev/null 2>&1 &
        done
    fi
else
    if [ $model_size -gt 10 ]; then
        python -m fastchat.serve.vllm_worker \
            --model-names $base_model \
            --model-path $base_model \
            --conv-template qwen-7b-chat \
            --enable-lora --lora-modules gen_sft=./results/iter_$iter/lora/gen_sft \
                ext_sft=./results/iter_$iter/lora/ext_sft \
            --controller-address $controller_address \
            --worker-address http://$my_ip:$port \
            --host $my_ip \
            --port $port > /dev/null 2>&1 &
    else
        for i in $(seq 0 $(($num_gpus - 1)))
        do
            echo "GPU $i is going to start"
            port=$((8080+$i))
            CUDA_VISIBLE_DEVICES=$i python -m fastchat.serve.vllm_worker \
                --model-names $base_model \
                --model-path $base_model \
                --conv-template qwen-7b-chat \
                --enable-lora --lora-modules gen_sft=./results/iter_$iter/lora/gen_sft \
                    ext_sft=./results/iter_$iter/lora/ext_sft \
                --controller-address $controller_address \
                --worker-address http://$my_ip:$port \
                --host $my_ip \
                --port $port > /dev/null 2>&1 &
        done
    fi
fi

echo "Waiting vllm server to start..."
