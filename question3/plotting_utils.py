"""Plotting utilities for optimization experiment visualization.

Author: Prabhav Singh
"""
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (momentum, trajectory) in enumerate(trajectories.items()):
        create_contour_plot(func, config, axes[i], f'Momentum = {momentum}')
        plot_trajectory(axes[i], trajectory, f'momentum={momentum}', colors[i])
        axes[i].legend()
    
    plt.suptitle(f'Momentum Comparison - {func.get_name()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def save_weight_decay_comparison_plot(func, trajectories_no_decay: Dict[float, List[tuple]],
                                    trajectories_with_decay: Dict[float, List[tuple]],
                                    config: Config, filename: str):
    """Save plot comparing with and without weight decay."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    momentum_values = [0.0, 0.9]
    colors = ['red', 'orange']
    
    for i, momentum in enumerate(momentum_values):
        create_contour_plot(func, config, axes[0, i], f'Momentum = {momentum} (No Weight Decay)')
        plot_trajectory(axes[0, i], trajectories_no_decay[momentum], 
                       f'momentum={momentum}', colors[i])
        axes[0, i].legend()
        
        create_contour_plot(func, config, axes[1, i], f'Momentum = {momentum} (Weight Decay = 0.1)')
        plot_trajectory(axes[1, i], trajectories_with_decay[momentum], 
                       f'momentum={momentum}', colors[i])
        axes[1, i].legend()
    
    plt.suptitle(f'Weight Decay Comparison - {func.get_name()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    plt.close()


def save_maximization_plot(func, trajectories: Dict[str, List[tuple]], 
                          config: Config, filename: str):
    """Save plot for maximization experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (label, trajectory) in enumerate(trajectories.items()):
        momentum = label.split('=')[1]
        create_contour_plot(func, config, axes[i], f'Momentum = {momentum} (maximize=True)')
        plot_trajectory(axes[i], trajectory, label, colors[i])
        axes[i].legend()
    
    plt.suptitle(f'Maximization Experiment - {func.get_name()}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/{filename}', dpi=300, bbox_inches='tight')
    plt.close()