#!/bin/bash
# Simple resume script - no auto-detection

PROJECT_DIR="/lambda/nfs/damienli-3/llm-trajectory-dynamics"
cd "$PROJECT_DIR"

# Kill old session
tmux kill-session -t resume 2>/dev/null || true

# Start new session
tmux new -d -s resume "python -u -m lmd.cli.collect \
    --data-root storage/datasets \
    --model-root storage/models \
    --output-dir storage/runs \
    --models Qwen2-7B-Instruct \
    --datasets math commonsenseqa theoremqa mmlu belebele hotpotqa \
    --max-new-tokens 2048 \
    --language en \
    --seed 42 \
    --verbose \
    --resume \
    --run-id run_20250913_064612_m1_d8 \
2>&1 | tee storage/logs/resume_$(date +%Y%m%d_%H%M%S).log"

echo "Started. Attach: tmux a -t resume"