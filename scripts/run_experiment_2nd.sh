#!/bin/bash

PARTITION="noniid-labeldir"   # Data partition strategy (e.g., homo, noniid-labeldir)
BETA=0.5             # Dirichlet distribution parameter
NOISE=0.1          # Noise value for homo

ALG="fedavg"       # Federated learning algorithm (e.g., fedavg, scaffold)

# Configuration Parameters
N_PARTIES=100        # Total number of clients
NUM_CLIENTS_P=20     # Number of regular clients
NUM_CLIENTS_Q=10      # Number of partial update clients

COMM_ROUNDS=100      # Number of communication rounds
EPOCHS=25            # Number of local epochs

# Model and Dataset Parameters
MODEL="simple-cnn"
DATASET="cifar10"
LR=0.01
BATCH_SIZE=64
RHO=0.9
DEVICE="cuda:0"
DATADIR="./data/"
LOGDIR_BASE="./logs"

INIT_SEED=0

# Proportion Calculations
P=$(bc -l <<< "$NUM_CLIENTS_P / $N_PARTIES")  # Fraction of regular clients
Q=$(bc -l <<< "$NUM_CLIENTS_Q / $N_PARTIES")  # Fraction of partial update clients
SAMPLE=$(bc -l <<< "$P + $Q")                # Total participation ratio

# Display Configuration
echo "========================================"
echo "Federated Learning Experiment"
echo "========================================"
echo "Total clients: $N_PARTIES"
echo "Algorithm: $ALG"
echo "Number of regular clients (P): $NUM_CLIENTS_P"
echo "Number of partial update clients (Q): $NUM_CLIENTS_Q"
echo "Proportion of regular clients (P): $P"
echo "Proportion of partial update clients (Q): $Q"
echo "Client participation rate (SAMPLE): $SAMPLE"
echo "========================================"



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