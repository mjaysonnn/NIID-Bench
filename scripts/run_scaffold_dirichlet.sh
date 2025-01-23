#!/bin/bash

# Define variables for common parameters
N_PARTIES=100
SAMPLE=0.2
BETA=0.5 # Dirichlet noise
COMM_ROUNDS=500
EPOCHS=5

MODEL="simple-cnn"
DATASET="cifar10"
LR=0.01
BATCH_SIZE=64
RHO=0.9
DEVICE="cuda:0"
DATADIR="./data/"
LOGDIR_BASE="./logs"
INIT_SEED=0
NOISE=0.0 # Homo noise

# Loop through partitions and algorithms
for PARTITION in noniid-labeldir # noniid-labeldir, homo -> Update NOISE if using homo
do
    for ALG in scaffold
    do
        # Define specific log directory for this algorithm
        LOGDIR="$LOGDIR_BASE/$MODEL/$ALG/$PARTITION/"

        # Run the experiment
        python experiments_baseline.py \
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