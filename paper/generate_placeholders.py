"""
Generate placeholder figures for the paper
Run this before compiling main.tex if actual experiment figures are not available
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def create_critical_depth_figure():
    """Create placeholder for critical depth analysis figure"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: R score vs compression depth
    ax = axes[0]
    depths = np.arange(2, 11)
    r_scores = [0.15, 0.22, 0.35, 0.48, 0.58, 0.65, 0.70, 0.73, 0.75]
    threshold = 0.45
    
    ax.plot(depths, r_scores, 'bo-', linewidth=2, markersize=8, label='R(k)')
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    ax.axvline(x=6, color='g', linestyle=':', alpha=0.7, label='k* = 6')
    ax.fill_between([2, 6], [0, 0], [0.8, 0.8], alpha=0.1, color='red', label='Collapse Regime')
    ax.fill_between([6, 10], [0, 0], [0.8, 0.8], alpha=0.1, color='green', label='Safe Regime')
    ax.set_xlabel('Compression Depth k', fontsize=12)
    ax.set_ylabel('Final Reasoning Score R(k)', fontsize=12)
    ax.set_title('(a) Reasoning Collapse Score', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 10.5)
    ax.set_ylim(0, 0.85)
    
    # Right: S score vs compression depth
    ax = axes[1]
    s_scores = [0.12, 0.25, 0.38, 0.52, 0.62, 0.70, 0.75, 0.78, 0.80]
    threshold_s = 0.50
    
    ax.plot(depths, s_scores, 'bs-', linewidth=2, markersize=8, label='S(k)')
    ax.axhline(y=threshold_s, color='r', linestyle='--', label=f'Threshold = {threshold_s}')
    ax.axvline(x=6, color='g', linestyle=':', alpha=0.7, label='k* = 6')
    ax.fill_between([2, 6], [0, 0], [0.9, 0.9], alpha=0.1, color='red')
    ax.fill_between([6, 10], [0, 0], [0.9, 0.9], alpha=0.1, color='green')
    ax.set_xlabel('Compression Depth k', fontsize=12)
    ax.set_ylabel('Final Evidence Survival S(k)', fontsize=12)
    ax.set_title('(b) Evidence Survival Score', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 10.5)
    ax.set_ylim(0, 0.9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'critical_depth_placeholder.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'critical_depth_placeholder.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created critical_depth_placeholder.pdf")


def create_tsne_figure():
    """Create placeholder for t-SNE visualization figure"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    np.random.seed(42)
    
    # (a) Full-LLM: distinct clusters
    ax = axes[0]
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i, color in enumerate(colors):
        angle = 2 * np.pi * i / len(colors)
        cx, cy = 3 * np.cos(angle), 3 * np.sin(angle)
        x = cx + np.random.randn(30) * 0.5
        y = cy + np.random.randn(30) * 0.5
        ax.scatter(x, y, c=color, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_title('(a) Full-LLM', fontsize=12)
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    
    # (b) Fixed-Mid: collapsed clusters
    ax = axes[1]
    for i, color in enumerate(colors):
        x = np.random.randn(30) * 0.8
        y = np.random.randn(30) * 0.8
        ax.scatter(x, y, c=color, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_title('(b) Fixed-Mid (k=6)', fontsize=12)
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    
    # (c) CARR: partially separated clusters
    ax = axes[2]
    for i, color in enumerate(colors):
        angle = 2 * np.pi * i / len(colors)
        cx, cy = 2.2 * np.cos(angle), 2.2 * np.sin(angle)
        x = cx + np.random.randn(30) * 0.6
        y = cy + np.random.randn(30) * 0.6
        ax.scatter(x, y, c=color, s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_title('(c) CARR', fontsize=12)
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    
    # Add legend
    patches = [mpatches.Patch(color=c, label=f'Intent {i+1}') for i, c in enumerate(colors)]
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.12, 0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tsne_placeholder.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'tsne_placeholder.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created tsne_placeholder.pdf")


def create_bio_placeholder():
    """Create placeholder for author biography photo"""
    fig, ax = plt.subplots(1, 1, figsize=(1, 1.25))
    ax.text(0.5, 0.5, 'Photo', ha='center', va='center', fontsize=10, 
            color='gray', transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor('#f0f0f0')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('gray')
    
    plt.savefig(FIGURES_DIR / 'bio_placeholder.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'bio_placeholder.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created bio_placeholder.pdf")


if __name__ == "__main__":
    print("Generating placeholder figures for paper...")
    create_critical_depth_figure()
    create_tsne_figure()
    create_bio_placeholder()
    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("\nTo compile the paper:")
    print("  cd paper")
    print("  pdflatex main.tex")
