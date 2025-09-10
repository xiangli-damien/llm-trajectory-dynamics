#!/usr/bin/env python3
"""Monitor data collection progress."""

import os
import json
import time
import psutil
from pathlib import Path
from datetime import datetime, timedelta

def get_tmux_sessions():
    """Get tmux sessions."""
    try:
        import subprocess
        result = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
        if result.returncode == 0:
            sessions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        sessions.append({
                            'name': parts[0],
                            'info': parts[1].strip()
                        })
            return sessions
    except Exception:
        pass
    return []

def get_latest_log():
    """Get latest log file."""
    log_dir = Path("logs")
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob("collection_*.log"))
    if not log_files:
        return None
    
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        return {
            'file': latest_log,
            'size_mb': latest_log.stat().st_size / 1024 / 1024,
            'lines': len(lines),
            'last_modified': datetime.fromtimestamp(latest_log.stat().st_mtime),
            'last_lines': lines[-20:] if lines else []
        }
    except Exception:
        return None

# GPU monitoring removed - use nvidia-smi insteadi

def get_output_info():
    """Get output directory information."""
    output_dir = Path("storage/runs")
    if not output_dir.exists():
        return None
    
    files = list(output_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    
    # Count model directories
    model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    
    # Find latest run directory
    latest_run = None
    if model_dirs:
        latest_run = max(model_dirs, key=lambda x: x.stat().st_mtime)
    
    # Check for status file
    status_info = None
    if latest_run:
        status_file = latest_run / "collection_status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status_info = json.load(f)
            except Exception:
                pass
    
    return {
        'path': output_dir,
        'files': len(files),
        'total_size_mb': total_size / 1024 / 1024,
        'model_dirs': len(model_dirs),
        'model_names': [d.name for d in model_dirs],
        'latest_run': latest_run.name if latest_run else None,
        'status': status_info
    }

def main():
    """Monitor collection progress."""
    print("Data Collection Monitor")
    print("=" * 50)
    
    # Check tmux sessions
    sessions = get_tmux_sessions()
    collection_session = None
    for session in sessions:
        if 'data_collection' in session['name']:
            collection_session = session
            break
    
    if collection_session:
        print(f"Collection session: {collection_session['name']}")
        print(f"Session info: {collection_session['info']}")
    else:
        print("No collection session found")
        print("Start collection with: ./scripts/start_collection.sh")
        return
    
    # Get log info
    log_info = get_latest_log()
    if log_info:
        print(f"\nLog file: {log_info['file']}")
        print(f"Size: {log_info['size_mb']:.1f} MB")
        print(f"Lines: {log_info['lines']}")
        print(f"Last modified: {log_info['last_modified']}")
        
        if log_info['last_lines']:
            print(f"\nLast 20 lines:")
            print("-" * 30)
            for line in log_info['last_lines']:
                print(line.rstrip())
    
    # GPU info removed - use 'nvidia-smi' command instead
    
    # Get output info
    output_info = get_output_info()
    if output_info:
        print(f"\nOutput directory: {output_info['path']}")
        print(f"Files: {output_info['files']}")
        print(f"Total size: {output_info['total_size_mb']:.1f} MB")
        print(f"Model directories: {output_info['model_dirs']}")
        if output_info['model_names']:
            print(f"Models: {', '.join(output_info['model_names'])}")
        
        # Show status if available
        if output_info['status']:
            status = output_info['status']
            print(f"\nLatest run: {output_info['latest_run']}")
            print(f"Status: {status.get('status', 'unknown')}")
            if status.get('status') == 'running':
                print(f"Progress: {status.get('completed_tasks', 0)}/{status.get('total_tasks', 0)} tasks")
                if status.get('current_model'):
                    print(f"Current model: {status['current_model']}")
                if status.get('current_dataset'):
                    print(f"Current dataset: {status['current_dataset']}")
            elif status.get('status') == 'completed':
                print(f"Completed: {status.get('completed_tasks', 0)}/{status.get('total_tasks', 0)} tasks")
                if status.get('total_time_seconds'):
                    print(f"Total time: {status['total_time_seconds']/60:.1f} minutes")
    
    print(f"\nCommands:")
    print(f"  Attach to session:    tmux attach -t data_collection")
    print(f"  Real-time log:        tail -f {log_info['file'] if log_info else 'logs/collection_*.log'}")
    print(f"  Kill session:         tmux kill-session -t data_collection")

if __name__ == "__main__":
    main()