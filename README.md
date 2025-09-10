# LLM Trajectory Dynamics

A data collection pipeline for analyzing LLM trajectory dynamics across multiple models and datasets.

## Features

- Multi-model support (Llama-3, Mistral, Qwen2)
- Dataset integration (MMLU, GSM8K, CommonsenseQA, etc.)
- Hidden state extraction and analysis
- Parquet and Zarr storage formats

## Quick Start

```bash
# Install dependencies
poetry install

# Run data collection
python scripts/collect_all_data.py
```

## Project Structure

- `lmd/` - Core pipeline modules
- `scripts/` - Collection and monitoring scripts
- `storage/` - Data and model storage
- `configs/` - Configuration files