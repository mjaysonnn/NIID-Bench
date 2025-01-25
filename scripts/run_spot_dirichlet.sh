#!/bin/bash

ALG="fedavg" # Change to other algorithms if needed

# Define variables for common parameters
DATASET="cifar10"
MODEL="simple-cnn"

N_PARTIES=100
NUM_P=20 # Number of regular clients
NUM_Q=15  # Number of partial update clients

PARTITION="noniid-labeldir" # Change to "homo" if needed
BETA=0.5 # Dirichlet noise

COMM_ROUNDS=250
LR_LIST=(0.001 0.1 1) # List of learning rates
EPOCHS=10

BATCH_SIZE=64
RHO=0.9
DEVICE="cuda:0"
DATADIR="./data/"
LOGDIR_BASE="./logs"
INIT_SEED=0

# Calculate proportions dynamically based on NUM_P and NUM_Q
P=$(echo "$NUM_P / $N_PARTIES" | bc -l) # Proportion of regular clients
Q=$(echo "$NUM_Q / $N_PARTIES" | bc -l) # Proportion of partial update clients
SAMPLE=$(echo "($NUM_P + $NUM_Q) / $N_PARTIES" | bc -l) # Total participation rate

echo "Total clients: $N_PARTIES"
echo "Number of regular clients: $NUM_P"
echo "Number of partial update clients: $NUM_Q"
echo "Proportion of regular clients (P): $P"
echo "Proportion of partial update clients (Q): $Q"
echo "Total participation rate (SAMPLE): $SAMPLE"
echo "Partition Type: $PARTITION"
echo "Algorithm: $ALG"

# Define specific log directory for this algorithm
LOGDIR="$LOGDIR_BASE/$MODEL/$ALG/$PARTITION/"

# Loop through learning rates
for LR in "${LR_LIST[@]}"; do
    echo "Running experiment with learning rate: $LR"

    # Run the experiment
    python experiments_spot.py \
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
        --sample=$SAMPLE \
        --init_seed=$INIT_SEED \
        --p=$P \
        --q=$Q
done