#!/bin/bash

model="vgg"
partition="noniid-labeldir"
beta=0.1

for alg in scaffold fedavg
do
    python experiments.py --model=$model \
        --dataset=cifar10 \
        --alg=$alg \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=10 \
        --n_parties=10 \
        --rho=0.9 \
        --mu=0.01 \
        --comm_round=100 \
        --partition=$partition \
        --beta=$beta \
        --device='cuda:0' \
        --datadir='./data/' \
        --logdir="./logs/$model/$alg/" \
        --noise=0 \
        --init_seed=0
done