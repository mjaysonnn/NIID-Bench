#!/bin/bash

# Communication rounds
COMM_ROUNDS=50

# Local epochs
EPOCHS=10

# Dirichlet distribution parameters
BETA=0.5 

# Client Variables
N_PARTIES=25  # Total number of clients
NUM_CLIENTS_P=20  # Number of regular clients
NUM_CLIENTS_Q=5   # Number of partial update clients

MODEL="simple-cnn"
DATASET="cifar10"
LR=0.01
BATCH_SIZE=64
RHO=0.9
DEVICE="cuda:0"
DATADIR="./data/"
LOGDIR_BASE="./logs"
NOISE=0
INIT_SEED=0

# Calculate proportions
P=$(echo "$NUM_CLIENTS_P / $N_PARTIES" | bc -l)  # Fraction of regular clients
Q=$(echo "$NUM_CLIENTS_Q / $N_PARTIES" | bc -l)  # Fraction of partial update clients
SAMPLE=$(echo "$P + $Q" | bc)  # Total participation ratio

echo "Total clients: $N_PARTIES"
echo "Number of regular clients (P): $NUM_CLIENTS_P"
echo "Number of partial update clients (Q): $NUM_CLIENTS_Q"
echo "Proportion of regular clients (P): $P"
echo "Proportion of partial update clients (Q): $Q"
echo "Client participation rate (SAMPLE): $SAMPLE"

# Loop through partitions and algorithms
for PARTITION in noniid-labeldir
do
    for ALG in fedavg
    do
        # Define specific log directory for this algorithm
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
            --init_seed=$INIT_SEED \
            --p $P \
            --q $Q
    done
done