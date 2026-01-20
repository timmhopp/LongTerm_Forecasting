#!/bin/bash
# Comprehensive training script for FusedODModel
# Trains both single-step and multi-step models

# Set paths 
ADJACENCY_PATH="/path/to/adjacency_matrix.csv"
DISTANCE_PATH="/path/to/distance_matrix.csv" 
TRIPS_TENSOR_PATH="/path/to/trips_tensor.pt"
CHECKPOINT_PATH="/path/to/model_checkpoint.pth"
OUTPUT_DIR="./training_outputs"

# Training parameters
EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-3
HIDDEN_SIZE=64

# Window parameters
W_LONG=144
W_SHORT=36
CHUNK_SIZE_SHORT=9
NUM_CHUNKS_SHORT=4

# Multi-step prediction horizons
PREDICTION_HORIZONS="1 36 144 432 1008"

# Single-Step Model Training
echo "[1/2] Training single-step model..."
echo "--------------------------------------"
echo "Configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Hidden size: $HIDDEN_SIZE"
echo "  - Long window: $W_LONG"
echo "  - Short window: $W_SHORT"
echo ""

python train_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $CHECKPOINT_PATH \
    --output-dir "${OUTPUT_DIR}/single_step" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT

if [ $? -eq 0 ]; then
    echo "✓ Single-step model training completed successfully!"
    SINGLE_STEP_MODEL=$CHECKPOINT_PATH
else
    echo "✗ Single-step model training failed!"
    exit 1
fi
echo ""

# Multi-Step Model Training
echo "[2/2] Training multi-step model..."
echo "--------------------------------------"
echo "Configuration:"
echo "  - Prediction horizons: $PREDICTION_HORIZONS"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo ""

# Create multi-step checkpoint path
MULTI_STEP_CHECKPOINT="${CHECKPOINT_PATH%.pth}_multi_step.pth"

python train_main.py \
    --adjacency-path $ADJACENCY_PATH \
    --distance-path $DISTANCE_PATH \
    --trips-tensor-path $TRIPS_TENSOR_PATH \
    --checkpoint-path $MULTI_STEP_CHECKPOINT \
    --output-dir "${OUTPUT_DIR}/multi_step" \
    --multi-step \
    --prediction-horizons $PREDICTION_HORIZONS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --hidden-size $HIDDEN_SIZE \
    --w-long $W_LONG \
    --w-short $W_SHORT \
    --chunk-size-short $CHUNK_SIZE_SHORT \
    --num-chunks-short $NUM_CHUNKS_SHORT

if [ $? -eq 0 ]; then
    echo "✓ Multi-step model training completed successfully!"
else
    echo "✗ Multi-step model training failed!"
    exit 1
fi
echo ""

# Summary

echo "Trained models saved to:"
echo "  - Single-step model: $CHECKPOINT_PATH"
echo "  - Multi-step model:  $MULTI_STEP_CHECKPOINT"
echo ""
echo "Training outputs saved to:"
echo "  - Single-step:  ${OUTPUT_DIR}/single_step/"
echo "  - Multi-step:   ${OUTPUT_DIR}/multi_step/"
