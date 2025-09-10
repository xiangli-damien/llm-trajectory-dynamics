#!/usr/bin/env python
import argparse
import os
from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace")
    parser.add_argument("--model-id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--local-dir", type=str, required=True, help="Local directory to save")
    parser.add_argument("--hf-token", type=str, help="HuggingFace API token")
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.hf_token or os.getenv("HF_TOKEN")
    
    # Create directory
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {args.model_id} to {local_dir}...")
    
    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=["*.json", "*.bin", "*.safetensors", "tokenizer*", "*model*"]
    )
    
    print(f"Successfully downloaded {args.model_id}")

if __name__ == "__main__":
    main()