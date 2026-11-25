#!/bin/bash
# run_all_realtime_eval.sh
# Comprehensive real-time evaluation for all models with multiple seeds

set -e  # Exit on error

# Configuration
RESULTS_DIR="results/realtime_eval"
SEEDS=(42 43 44 45 46)
N_RUNS=${#SEEDS[@]}
STREAM_DURATION=20
LATENCY_ITERATIONS=300
OUT_CHANNELS=2  # PhysiQ binary classification

# Create results directory
mkdir -p $RESULTS_DIR

echo "=============================================="
echo "Real-Time Evaluation Benchmark"
echo "=============================================="
echo "Number of runs per model: $N_RUNS"
echo "Seeds: ${SEEDS[@]}"
echo "Results directory: $RESULTS_DIR"
echo "Stream duration: ${STREAM_DURATION}s"
echo "Latency iterations: $LATENCY_ITERATIONS"
echo ""

# Define models and their checkpoint paths
declare -A MODELS
MODELS["caex"]="artifacts/caex-physiq-42:v1/caex-physiq-42.pth"
MODELS["ex"]="artifacts/ex-physiq-43:v0/ex-physiq-43.pth"
MODELS["unet"]="artifacts/unet-physiq-43:v0/unet-physiq-43.pth"
MODELS["transformer"]="artifacts/transformer-physiq-43:v0/transformer-physiq-43.pth"
MODELS["segmenter"]="artifacts/segmenter-physiq-43:v0/segmenter-physiq-43.pth"
MODELS["cnn"]="artifacts/cnn-physiq-43:v0/cnn-physiq-43.pth"

# Track success/failure
declare -A SUCCESS_COUNT

# Run evaluation for each model
for MODEL in caex ex unet transformer segmenter cnn; do
    MODEL_PATH="${MODELS[$MODEL]}"
    SUCCESS_COUNT[$MODEL]=0
    
    echo ""
    echo "=============================================="
    echo "Evaluating: $MODEL"
    echo "Checkpoint: $MODEL_PATH"
    echo "=============================================="
    
    # Check if checkpoint exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "WARNING: Checkpoint not found: $MODEL_PATH"
        echo "Skipping $MODEL..."
        continue
    fi
    
    # Run N times with different seeds
    for i in $(seq 0 $((N_RUNS - 1))); do
        SEED=${SEEDS[$i]}
        echo ""
        echo "--- Run $((i + 1))/$N_RUNS (seed=$SEED) ---"
        
        if python eval_realtime.py \
            --model $MODEL \
            --model_path "$MODEL_PATH" \
            --mode full \
            --device cuda \
            --out_channels $OUT_CHANNELS \
            --stream_duration $STREAM_DURATION \
            --latency_iterations $LATENCY_ITERATIONS \
            --seed $SEED \
            --json_output "$RESULTS_DIR/${MODEL}_seed${SEED}.json" \
            --output "$RESULTS_DIR/${MODEL}_seed${SEED}.md" 2>&1; then
            SUCCESS_COUNT[$MODEL]=$((${SUCCESS_COUNT[$MODEL]} + 1))
        else
            echo "ERROR: Evaluation failed for $MODEL with seed $SEED"
        fi
    done
    
    echo ""
    echo "$MODEL completed: ${SUCCESS_COUNT[$MODEL]}/$N_RUNS successful"
done

echo ""
echo "=============================================="
echo "Evaluation Summary"
echo "=============================================="
for MODEL in caex ex unet transformer segmenter cnn; do
    echo "$MODEL: ${SUCCESS_COUNT[$MODEL]:-0}/$N_RUNS runs"
done

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="
