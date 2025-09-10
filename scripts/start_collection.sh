#!/bin/bash
# Start data collection in tmux session

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
SESSION_NAME="data_collection"
LOG_DIR="logs"
OUTPUT_DIR="storage/runs"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/collection_${TIMESTAMP}.log"

echo "Starting data collection in tmux session..."
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
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR" \
    "python -u scripts/collect_all_data.py \
        --data_root storage/datasets \
        --model_root storage/models \
        --output_dir storage/runs \
        --models Qwen2-7B-Instruct \
        --datasets gsm8k mmlu mgsm math commonsenseqa hotpotqa theoremqa \
        --full_collection \
        --max_new_tokens 2048 \
        --top_p 1.0 \
        --language en \
        --seed 42 \
        --verbose \
    2>&1 | tee $LOG_FILE"

# Wait a moment for session to start
sleep 2

# Check if session is running
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' started successfully"
    echo ""
    echo "Useful commands:"
    echo "  Attach to session:    tmux attach -t $SESSION_NAME"
    echo "  Detach from session:  Ctrl+B, then D"
    echo "  Kill session:         tmux kill-session -t $SESSION_NAME"
    echo "  List sessions:        tmux list-sessions"
    echo ""
    echo "Monitor progress:"
    echo "  Real-time log:        tail -f $LOG_FILE"
    echo "  Check status:         tmux list-sessions"
    echo ""
    echo "Output location: $OUTPUT_DIR"
else
    echo "Failed to start session"
    exit 1
fi