#!/bin/bash

for partition in homo
do
    for alg in scaffold
    do
        python experiments.py --model=resnet \
            --dataset=cifar10 \
            --alg=$alg \
            --lr=0.01 \
            --batch-size=64 \
            --epochs=10 \
            --n_parties=10 \
            --rho=0.9 \
            --comm_round=100 \
            --partition=$partition \
            --beta=0.5 \
            --device='cuda:0' \
            --datadir='./data/' \
            --logdir="./logs/resnet/$alg/" \
            --noise=0.1 \
            --sample=1.0 \
            --init_seed=0
    done
done