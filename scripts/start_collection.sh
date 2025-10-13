#!/bin/bash
# Simple collection script

PROJECT_DIR="/lambda/nfs/damienli-3/llm-trajectory-dynamics"
cd "$PROJECT_DIR"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM="false"

SESSION="collect_$(date +%H%M)"
LOG="storage/logs/run_$(date +%Y%m%d_%H%M%S).log"

mkdir -p storage/logs

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new -d -s "$SESSION" \
    "python -u -m lmd.cli.collect \
        --data-root storage/datasets \
        --model-root storage/models \
        --output-dir storage/runs \
        --models Qwen2-7B-Instruct \
        --datasets mmlu belebele commonsenseqa hotpotqa theoremqa math \
        --max-new-tokens 2048 \
        --language en \
        --seed 42 \
        --flush-every 50 \
        --timeout-s 2400 \
        --zarr-chunk-t 64 \
        --verbose \
    2>&1 | tee $LOG"

echo "Started. Attach: tmux a -t $SESSION"
echo "Log: $LOG"