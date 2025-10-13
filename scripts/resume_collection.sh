#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/lambda/nfs/damienli-3/llm-trajectory-dynamics"
RUN_ID="run_20250913_064612_m1_d8"
SESSION="resume_math_$(date +%m%d_%H%M)"
LOG_DIR="$PROJECT_DIR/storage/logs"
mkdir -p "$LOG_DIR"

# Suggested allocator/environment, further reducing fragmentation and log noise
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_DISABLE_TELEMETRY=1
ulimit -n 65535 || true

cd "$PROJECT_DIR"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new -d -s "$SESSION" "python -u -m lmd.cli.collect \
  --data-root storage/datasets \
  --model-root storage/models \
  --output-dir storage/runs \
  --models Qwen2-7B-Instruct \
  --datasets math \
  --max-new-tokens 2048 \
  --language en \
  --seed 42 \
  --flush-every 80 \
  --timeout-s 2400 \
  --zarr-chunk-t 64 \
  --resume \
  --run-id $RUN_ID \
  --skip-indices '' \
  --verbose \
  2>&1 | tee $LOG_DIR/resume_math_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "Started. Attach with: tmux a -t $SESSION"
