#!/bin/bash

# Define variables for common parameters
MODEL="simple-cnn"
DATASET="cifar10"
EPOCHS=10
N_PARTIES=100
COMM_ROUNDS=500
SAMPLE=0.1
NOISE=0.1 # Homo noise
BETA=0.5 # Dirichlet noise

LR=0.01
BATCH_SIZE=64
RHO=0.9
DEVICE="cuda:0"
DATADIR="./data/"
LOGDIR_BASE="./logs"
INIT_SEED=0



# Loop through partitions and algorithms
for PARTITION in homo # noniid-labeldir, homo -> Update NOISE if using homo
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