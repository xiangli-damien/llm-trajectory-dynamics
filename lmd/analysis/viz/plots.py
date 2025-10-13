import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from pathlib import Path
from ..config import PLOT_COLORS
from .style import setup_plot_style

def distribution(labels: np.ndarray, scores: Dict[str, np.ndarray], 
                bins: int = 30, title: str = "Score Distributions",
                savepath: Optional[Path] = None):
    setup_plot_style()
    
    n = len(scores)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if n == 1:
        axes = [axes]
    else:
        axes = np.atleast_1d(axes).ravel()
    
    for ax, (name, s) in zip(axes, scores.items()):
        correct_scores = s[labels == 1]
        incorrect_scores = s[labels == 0]
        
        ax.hist(correct_scores, bins=bins, alpha=0.5, label='Correct', 
               color=PLOT_COLORS["correct"], density=True)
        ax.hist(incorrect_scores, bins=bins, alpha=0.5, label='Incorrect', 
               color=PLOT_COLORS["incorrect"], density=True)
        
        ax.axvline(np.median(correct_scores), color=PLOT_COLORS["correct"], 
                  linestyle='--', alpha=0.7)
        ax.axvline(np.median(incorrect_scores), color=PLOT_COLORS["incorrect"], 
                  linestyle='--', alpha=0.7)
        
        ax.set_title(name)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for ax in axes[len(scores):]:
        ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    
    plt.show()

def auroc_bar(evaluations: Dict[str, Dict[str, float]], 
              top_k: Optional[int] = None,
              title: str = "AUROC Comparison",
              savepath: Optional[Path] = None):
    setup_plot_style()
    
    items = [(k, v["auroc"]) for k, v in evaluations.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    
    if top_k:
        items = items[:top_k]
    
    names = [k for k, _ in items]
    values = [v for _, v in items]
    
    fig, ax = plt.subplots(figsize=(max(8, 0.4*len(items)), 5))
    
    bars = ax.bar(range(len(items)), values, color=PLOT_COLORS["primary"])
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('AUROC')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    
    plt.show()

def layer_auroc_curve(layer_evals: Dict[str, List[Dict]], 
                     title: str = "Layerwise AUROC",
                     savepath: Optional[Path] = None):
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metric_key, eval_list in layer_evals.items():
        aurocs = [e["auroc"] for e in eval_list]
        layers = list(range(len(aurocs)))
        ax.plot(layers, aurocs, marker='o', label=metric_key, linewidth=2)
    
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('AUROC')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    
    plt.show()

def layer_gap_profile(labels: np.ndarray, per_layer_values: np.ndarray,
                      metric_key: str, stat: str = "mean",
                      title: Optional[str] = None,
                      savepath: Optional[Path] = None):
    setup_plot_style()
    
    correct_mask = labels == 1
    incorrect_mask = labels == 0
    
    N, L = per_layer_values.shape
    layers = list(range(L))
    
    if stat == "mean":
        correct_stat = np.mean(per_layer_values[correct_mask], axis=0)
        incorrect_stat = np.mean(per_layer_values[incorrect_mask], axis=0)
    elif stat == "median":
        correct_stat = np.median(per_layer_values[correct_mask], axis=0)
        incorrect_stat = np.median(per_layer_values[incorrect_mask], axis=0)
    else:
        raise ValueError(f"Unknown stat: {stat}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(layers, correct_stat, marker='o', label='Correct',
             color=PLOT_COLORS["correct"], linewidth=2)
    ax1.plot(layers, incorrect_stat, marker='s', label='Incorrect',
             color=PLOT_COLORS["incorrect"], linewidth=2)
    ax1.set_ylabel(f'{stat.capitalize()} {metric_key}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if title:
        ax1.set_title(title)
    
    gap = correct_stat - incorrect_stat
    ax2.plot(layers, gap, marker='^', color=PLOT_COLORS["accent"], linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Gap (Correct - Incorrect)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    
    plt.show()