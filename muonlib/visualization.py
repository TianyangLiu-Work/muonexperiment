"""Visualization utilities for Muon experiment results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# Style configuration
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
})

ALGORITHM_COLORS = {
    'Muon-Exact': '#2196F3', 'Muon': '#2196F3',
    'SGD': '#FF9800',
    'Adam': '#4CAF50', 'RMSprop': '#F44336',
    'Momentum-SGD': '#9C27B0', 'Momentum': '#9C27B0',
}

ALGORITHM_LINESTYLES = {
    'Muon-Exact': '-', 'Muon-RandSVD': '--', 'Muon-Trunc': ':',
    'SGD': '-', 'Adam': '-', 'RMSprop': '-', 'Momentum-SGD': '-',
}


def plot_convergence(trajectories, title=None, xlabel='Iteration', 
                     ylabel='Loss', log_scale=True, save_path=None):
    """Plot convergence curves for multiple algorithms/configurations.
    
    trajectories: {label: np.array of losses}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for label, losses in trajectories.items():
        color = ALGORITHM_COLORS.get(label.split('_')[0], None)
        ls = ALGORITHM_LINESTYLES.get(label.split('_')[0], '-')
        ax.plot(losses, label=label, color=color, linestyle=ls, alpha=0.8, linewidth=1.5)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_heatmap(data, row_labels, col_labels, title=None, 
                 xlabel='', ylabel='', cmap='RdYlBu_r', save_path=None):
    """Plot a heatmap (e.g., rank ratio x dimension -> convergence advantage)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)
    
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=8)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_boxplot(data_dict, title=None, ylabel='', save_path=None):
    """Compare distributions with boxplots.
    
    data_dict: {label: list of values}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    
    bp = ax.boxplot(values, labels=labels, patch_artist=True)
    
    for i, (patch, label) in enumerate(zip(bp['boxes'], labels)):
        color = ALGORITHM_COLORS.get(label.split('_')[0], '#CCCCCC')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_pareto_frontier(x_vals, y_vals, labels, xlabel='', ylabel='', 
                          title=None, save_path=None):
    """Plot Pareto frontier (e.g., accuracy vs wall-clock time)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, label in enumerate(labels):
        color = ALGORITHM_COLORS.get(label.split('_')[0], None)
        ax.scatter(x_vals[i], y_vals[i], label=label, color=color, s=50, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_convergence_with_ci(trajectories, cis, title=None, save_path=None):
    """Plot convergence curves with confidence interval bands."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for label, losses in trajectories.items():
        color = ALGORITHM_COLORS.get(label.split('_')[0], None)
        ls = ALGORITHM_LINESTYLES.get(label.split('_')[0], '-')
        
        mean = np.mean(np.array(losses), axis=0)
        ax.plot(mean, label=label, color=color, linestyle=ls, linewidth=1.5)
        
        if label in cis:
            ci = cis[label]
            ax.fill_between(range(len(mean)), mean - ci, mean + ci,
                            color=color, alpha=0.15)
    
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig
