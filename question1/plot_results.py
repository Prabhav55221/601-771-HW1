"""
Plotting utilities for self-attention profiling results
Uses seaborn for error band plots as requested
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class ResultsPlotter:
    """
    Create publication-quality plots with error bars for profiling results
    """
    
    def __init__(self, results_dir="results/", style="whitegrid"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set seaborn style
        sns.set_style(style)
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11
        })
    
    def plot_complexity_trends(self, results_df, save=True):
        """
        Plot FLOPS, memory usage, and timing trends with error bars
        
        Args:
            results_df: DataFrame with columns ['seq_len', 'attention_type', 'device', 
                                              'flops_mean', 'flops_std_error',
                                              'memory_mean', 'memory_std_error',
                                              'time_mean', 'time_std_error']
            save: whether to save plots to files
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: FLOPS vs Sequence Length
        self._plot_metric(results_df, 'flops_mean', 'flops_std_error', 
                         'Sequence Length', 'FLOPs', 
                         'Computational Complexity (FLOPs)', axes[0])
        
        # Plot 2: Memory Usage vs Sequence Length  
        self._plot_metric(results_df, 'memory_mean', 'memory_std_error',
                         'Sequence Length', 'Memory Usage (MB)',
                         'Memory Usage', axes[1])
        
        # Plot 3: Wall Clock Time vs Sequence Length
        self._plot_metric(results_df, 'time_mean', 'time_std_error',
                         'Sequence Length', 'Time (seconds)',
                         'Wall Clock Time', axes[2])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.results_dir / "attention_complexity_trends.png", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.results_dir / "attention_complexity_trends.pdf", 
                       bbox_inches='tight')
        
        return fig
    
    def _plot_metric(self, df, y_col, error_col, x_label, y_label, title, ax):
        """
        Plot a single metric with error bars using seaborn
        """
        # Create separate plots for each combination of attention_type and device
        for attention_type in df['attention_type'].unique():
            for device in df['device'].unique():
                subset = df[(df['attention_type'] == attention_type) & 
                          (df['device'] == device)]
                
                if len(subset) > 0:
                    label = f"{attention_type} ({device.upper()})"
                    
                    # Use seaborn lineplot with error bars
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
    
    def plot_scaling_analysis(self, results_df, save=True):
        """
        Create separate detailed plots for scaling analysis
        """
        metrics = [
            ('flops_mean', 'flops_std_error', 'FLOPs', 'Computational Complexity'),
            ('memory_mean', 'memory_std_error', 'Memory Usage (MB)', 'Memory Usage'),
            ('time_mean', 'time_std_error', 'Time (seconds)', 'Wall Clock Time')
        ]
        
        for mean_col, error_col, y_label, title in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot each combination
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
            
            if save:
                filename = title.lower().replace(' ', '_')
                plt.savefig(self.results_dir / f"{filename}_scaling.png", 
                           dpi=300, bbox_inches='tight')
                plt.savefig(self.results_dir / f"{filename}_scaling.pdf", 
                           bbox_inches='tight')
            
            plt.show()
    
    def create_comparison_table(self, results_df, save=True):
        """
        Create a summary table of results
        """
        # Pivot table for better readability
        summary_tables = {}
        
        for metric in ['flops_mean', 'memory_mean', 'time_mean']:
            pivot = results_df.pivot_table(
                values=metric, 
                index='seq_len', 
                columns=['attention_type', 'device'],
                aggfunc='mean'
            )
            summary_tables[metric] = pivot
            
            if save:
                metric_name = metric.replace('_mean', '')
                pivot.to_csv(self.results_dir / f"{metric_name}_summary.csv")
        
        return summary_tables
    
    def plot_complexity_comparison(self, results_df, save=True):
        """
        Create a comparison plot showing theoretical vs empirical scaling
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        seq_lengths = sorted(results_df['seq_len'].unique())
        
        # Theoretical complexity curves
        x_theory = np.logspace(1, 4, 100)  # 10 to 10,000
        
        for idx, attention_type in enumerate(['single_head', 'multi_head']):
            subset = results_df[results_df['attention_type'] == attention_type]
            
            if len(subset) == 0:
                continue
                
            ax = axes[idx * 2]
            ax2 = axes[idx * 2 + 1]
            
            # Plot empirical FLOPS
            for device in subset['device'].unique():
                device_data = subset[subset['device'] == device]
                ax.errorbar(device_data['seq_len'], device_data['flops_mean'],
                          yerr=device_data['flops_std_error'],
                          label=f'{device.upper()} (empirical)', marker='o')
            
            # Plot theoretical scaling (O(n²d) for attention computation)
            d_model = 768  # from config
            if attention_type == 'single_head':
                theory_flops = 3 * x_theory * d_model**2 + 2 * x_theory**2 * d_model
            else:  # multi_head
                theory_flops = 4 * x_theory * d_model**2 + 2 * x_theory**2 * d_model
            
            ax.plot(x_theory, theory_flops, 'k--', label='Theoretical', alpha=0.7)
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('FLOPs')
            ax.set_title(f'{attention_type.replace("_", "-").title()} - FLOPS Scaling')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot empirical timing
            for device in subset['device'].unique():
                device_data = subset[subset['device'] == device]
                ax2.errorbar(device_data['seq_len'], device_data['time_mean'],
                           yerr=device_data['time_std_error'],
                           label=f'{device.upper()}', marker='s')
            
            ax2.set_xlabel('Sequence Length')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title(f'{attention_type.replace("_", "-").title()} - Timing')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.results_dir / "complexity_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.results_dir / "complexity_comparison.pdf", 
                       bbox_inches='tight')
        
        return fig