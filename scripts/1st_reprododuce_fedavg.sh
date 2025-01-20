#!/bin/bash

for partition in noniid-labeldir
do
    for alg in fedavg
    do
        python experiments.py --model=simple-cnn \
            --dataset=cifar10 \
            --alg=$alg \
            --lr=0.01 \
            --batch-size=64 \
            --epochs=10 \
            --n_parties=10 \
            --rho=0.9 \
            --comm_round=50 \
            --partition=$partition \
            --beta=0.5 \
            --device='cuda:0' \
            --datadir='./data/' \
            --logdir="./logs/$alg/" \
            --noise=0 \
            --sample=1.0 \
            --init_seed=0
    done
done