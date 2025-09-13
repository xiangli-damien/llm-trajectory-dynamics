#!/bin/bash
# Full data collection script for LLM trajectory dynamics

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
SESSION_NAME="llm_collection"
LOG_DIR="storage/logs"
OUTPUT_DIR="storage/runs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/collection_${TIMESTAMP}.log"

echo "Starting LLM data collection in tmux session..."
echo "Session name: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists"
    echo "Use 'tmux attach -t $SESSION_NAME' to attach"
    echo "Use 'tmux kill-session -t $SESSION_NAME' to kill existing session"
    exit 1
fi

# Create new tmux session and run collection
# Using the main CLI module with proper flags
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR" \
    "python -u -m lmd.cli.collect \
        --data-root storage/datasets \
        --model-root storage/models \
        --output-dir storage/runs \
        --models Qwen2-7B-Instruct \
        --datasets mgsm gsm8k math commonsenseqa theoremqa mmlu belebele hotpotqa \
        --max-new-tokens 2048 \
        --language en \
        --seed 42 \
        --verbose \
    2>&1 | tee $LOG_FILE"

# Wait a moment for session to start
sleep 2

# Check if session is running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "Session '$SESSION_NAME' started successfully"
    echo "=========================================="
    echo ""
    echo "Collection parameters:"
    echo "  Models: Qwen2-7B-Instruct"
    echo "  Datasets: mgsm, gsm8k, math, commonsenseqa, mmlu, theoremqa, hotpotqa"
    echo ""
    echo "  Attach to session:    tmux attach -t $SESSION_NAME"
    echo ""
    echo "Output location: $OUTPUT_DIR"
    echo "Log location: $LOG_FILE"
else
    echo "Failed to start session"
    exit 1
fi
