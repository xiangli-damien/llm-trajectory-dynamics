import os

NUMCODECS_THREADS = min(20, os.cpu_count())
os.environ.setdefault("NUMCODECS_THREADS", str(NUMCODECS_THREADS))

DEFAULT_CACHE_SIZE_GB = 2.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_JOBS = -1

LABEL_CORRECT = 1
LABEL_INCORRECT = 0

PLOT_COLORS = {
    "correct": "#1565C0",
    "incorrect": "#C62828",
    "neutral": "#616161",
    "primary": "#1565C0",
    "secondary": "#6A1B9A",
    "accent": "#44AA99",
    "grid": "#E5E7EB",
}