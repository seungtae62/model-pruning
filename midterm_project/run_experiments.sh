#!/bin/bash
# GEV6152 Midterm Project - Neural Network Pruning Experiments
# Author: seungtae62
# GitHub: https://github.com/seungtae62/model-pruning

set -e  # Exit on error

echo "=========================================="
echo "GEV6152 Midterm Project - Pruning Experiments"
echo "=========================================="
echo ""

# Configuration
SEEDS=(42 43 44)
DEVICE="cuda"  # Change to "cpu" if no GPU available

if [ ! -d "train" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

echo "Step 1: Training Dense Baseline Model"
echo "--------------------------------------"
python train/train_dense.py \
    --seed 42 \
    --epochs 200 \
    --device ${DEVICE}
echo "✓ Dense baseline training completed"
echo ""

echo "Step 2: Training Lottery Ticket Hypothesis"
echo "-------------------------------------------"
for seed in "${SEEDS[@]}"; do
    echo "Training Lottery Ticket with seed ${seed}..."
    python train/train_lottery.py \
        --seed ${seed} \
        --num_rounds 10 \
        --prune_rate 0.2 \
        --epochs 200 \
        --device ${DEVICE}
    echo "✓ Lottery Ticket seed ${seed} completed"
done
echo ""

echo "Step 3: Training Magnitude Pruning"
echo "-----------------------------------"
BASELINE_PATH="checkpoints/resnet18/resnet18_baseline_best.pth"

if [ ! -f "${BASELINE_PATH}" ]; then
    echo "Error: Baseline model not found at ${BASELINE_PATH}"
    echo "Please train the dense baseline first"
    exit 1
fi

for seed in "${SEEDS[@]}"; do
    echo "Training Magnitude Pruning with seed ${seed}..."
    python train/train_magnitude.py \
        --seed ${seed} \
        --pretrained_path ${BASELINE_PATH} \
        --num_iterations 7 \
        --quality_param 1.0 \
        --finetune_lr 0.01 \
        --finetune_epochs 100 \
        --device ${DEVICE}
    echo "✓ Magnitude Pruning seed ${seed} completed"
done
echo ""
