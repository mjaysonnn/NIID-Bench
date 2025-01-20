#!/bin/bash

model="simple-cnn"
partition="noniid-labeldir"
beta=0.5

for alg in scaffold
do
    python experiments.py --model=$model \
        --dataset=cifar10 \
        --alg=$alg \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=10 \
        --n_parties=10 \
        --rho=0.9 \
        --comm_round=50 \
        --partition=$partition \
        --beta=$beta \
        --device='cuda:0' \
        --datadir='./data/' \
        --logdir="./logs/$model/$alg/" \
        --noise=0 \
        --sample=1.0 \
        --init_seed=0
done