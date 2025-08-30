import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os

from config import Config


def create_contour_plot(func, config: Config, ax, title: str):
    """Create contour plot for the optimization function."""
    x = np.linspace(-config.plot_range, config.plot_range, 100)
    y = np.linspace(-config.plot_range, config.plot_range, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i, j], Y[i, j])
    
    contour = ax.contour(X, Y, Z, levels=config.contour_levels, alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-config.plot_range, config.plot_range)
    ax.set_ylim(-config.plot_range, config.plot_range)
    
    return contour


def plot_trajectory(ax, trajectory: List[tuple], label: str, color: str, alpha: float = 0.8):
    """Plot optimization trajectory on existing axes."""
    x_coords = [point[0] for point in trajectory]
    y_coords = [point[1] for point in trajectory]
    
    ax.plot(x_coords, y_coords, 'o-', color=color, alpha=alpha, 
            markersize=3, linewidth=1.5, label=label)
    ax.plot(x_coords[0], y_coords[0], 's', color=color, markersize=8, alpha=0.9)
    ax.plot(x_coords[-1], y_coords[-1], '^', color=color, markersize=8, alpha=0.9)


def save_momentum_comparison_plot(func, trajectories: Dict[float, List[tuple]], 
                                config: Config, filename: str):
    """Save plot comparing different momentum values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    create_contour_plot(func, config, ax, 
                       f'Momentum Comparison - {func.get_name()}')
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (momentum, trajectory) in enumerate(trajectories.items()):
        plot_trajectory(ax, trajectory, f'momentum={momentum}', colors[i])
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def save_weight_decay_comparison_plot(func, trajectories_no_decay: Dict[float, List[tuple]],
                                    trajectories_with_decay: Dict[float, List[tuple]],
                                    config: Config, filename: str):
    """Save plot comparing with and without weight decay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figure_size)
    
    create_contour_plot(func, config, ax1, 'Without Weight Decay')
    create_contour_plot(func, config, ax2, 'With Weight Decay (0.1)')
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, (momentum, trajectory) in enumerate(trajectories_no_decay.items()):
        plot_trajectory(ax1, trajectory, f'momentum={momentum}', colors[i])
    
    for i, (momentum, trajectory) in enumerate(trajectories_with_decay.items()):
        plot_trajectory(ax2, trajectory, f'momentum={momentum}', colors[i])
    
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def save_maximization_plot(func, trajectories: Dict[str, List[tuple]], 
                          config: Config, filename: str):
    """Save plot for maximization experiment."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    create_contour_plot(func, config, ax, 
                       f'Maximization with maximize=True - {func.get_name()}')
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (label, trajectory) in enumerate(trajectories.items()):
        plot_trajectory(ax, trajectory, label, colors[i])
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    plt.close()