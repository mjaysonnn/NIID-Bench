#!/bin/bash

for alg in scaffold
do
    python experiments.py --model=vgg \
        --dataset=cifar10 \
        --alg=$alg \
        --lr=0.01 \
        --batch-size=64 \
        --epochs=10 \
        --n_parties=10 \
        --rho=0.9 \
        --mu=0.01 \
        --comm_round=100 \
        --partition=noniid-labeldir \
        --beta=0.1 \
        --device='cuda:0' \
        --datadir='./data/' \
        --logdir="./logs/vgg/$alg/" \
        --noise=0 \
        --init_seed=0
done