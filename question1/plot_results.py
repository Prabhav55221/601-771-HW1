import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_complexity_trends(results_df):
    """Plot FLOPS, memory usage, and timing trends with error bars"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    _plot_metric(results_df, 'flops_mean', 'flops_std_error', 
                 'Sequence Length', 'FLOPs', 
                 'Computational Complexity (FLOPs)', axes[0])
    
    _plot_metric(results_df, 'memory_mean', 'memory_std_error',
                 'Sequence Length', 'Memory Usage (MB)',
                 'Memory Usage', axes[1])
    
    _plot_metric(results_df, 'time_mean', 'time_std_error',
                 'Sequence Length', 'Time (seconds)',
                 'Wall Clock Time', axes[2])
    
    plt.tight_layout()
    plt.savefig('attention_complexity_trends.png', dpi=300, bbox_inches='tight')
    plt.savefig('attention_complexity_trends.pdf', bbox_inches='tight')
    plt.close()


def _plot_metric(df, y_col, error_col, x_label, y_label, title, ax):
    """Plot a single metric with error bars"""
    for attention_type in df['attention_type'].unique():
        for device in df['device'].unique():
            subset = df[(df['attention_type'] == attention_type) & 
                      (df['device'] == device)]
            
            if len(subset) > 0:
                label = f"{attention_type} ({device.upper()})"
                
                ax.errorbar(subset['seq_len'], subset[y_col], 
                          yerr=subset[error_col],
                          label=label, marker='o', linewidth=2, 
                          markersize=6, capsize=5, capthick=2)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_scaling_analysis(results_df):
    """Create separate detailed plots for scaling analysis"""
    metrics = [
        ('flops_mean', 'flops_std_error', 'FLOPs', 'Computational Complexity'),
        ('memory_mean', 'memory_std_error', 'Memory Usage (MB)', 'Memory Usage'),
        ('time_mean', 'time_std_error', 'Time (seconds)', 'Wall Clock Time')
    ]
    
    for mean_col, error_col, y_label, title in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'v']
        
        i = 0
        for attention_type in sorted(results_df['attention_type'].unique()):
            for device in sorted(results_df['device'].unique()):
                subset = results_df[(results_df['attention_type'] == attention_type) & 
                                  (results_df['device'] == device)]
                
                if len(subset) > 0:
                    label = f"{attention_type} ({device.upper()})"
                    
                    ax.errorbar(subset['seq_len'], subset[mean_col], 
                              yerr=subset[error_col],
                              label=label, marker=markers[i % len(markers)], 
                              color=colors[i % len(colors)],
                              linewidth=2, markersize=8, capsize=5, capthick=2)
                    i += 1
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(y_label)
        ax.set_title(f'{title} vs Sequence Length')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        filename = title.lower().replace(' ', '_')
        plt.savefig(f'{filename}_scaling.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{filename}_scaling.pdf', bbox_inches='tight')
        plt.close()