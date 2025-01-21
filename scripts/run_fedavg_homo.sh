#!/bin/bash

# Define variables for common parameters
MODEL="simple-cnn"
DATASET="cifar10"
LR=0.01
BATCH_SIZE=64
EPOCHS=10
N_PARTIES=10
RHO=0.9
COMM_ROUNDS=50
BETA=0.5
DEVICE="cuda:0"
DATADIR="./data/"
LOGDIR_BASE="./logs"
NOISE=0.1
SAMPLE=1.0
INIT_SEED=0

# Loop through partitions and algorithms
for PARTITION in homo
do
    for ALG in fedavg
    do
        # Define log directory dynamically
        LOGDIR="$LOGDIR_BASE/$MODEL/$ALG/$PARTITION/"

        # Run the experiment
        python experiments.py \
            --model=$MODEL \
            --dataset=$DATASET \
            --alg=$ALG \
            --lr=$LR \
            --batch-size=$BATCH_SIZE \
            --epochs=$EPOCHS \
            --n_parties=$N_PARTIES \
            --rho=$RHO \
            --comm_round=$COMM_ROUNDS \
            --partition=$PARTITION \
            --beta=$BETA \
            --device=$DEVICE \
            --datadir=$DATADIR \
            --logdir=$LOGDIR \
            --noise=$NOISE \
            --sample=$SAMPLE \
            --init_seed=$INIT_SEED
    done
done