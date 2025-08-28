"""
Main profiler for self-attention performance analysis
Measures FLOPS, memory usage, and wall clock time for different sequence lengths
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys
from tqdm import tqdm

from config import Config
from attention_layers import SingleHeadAttention, MultiHeadAttention
from profiling_utils import FLOPSCounter, MemoryProfiler, Timer, compute_statistics
from plot_results import ResultsPlotter


class SelfAttentionProfiler:
    """
    Comprehensive profiler for self-attention mechanisms
    """
    
    def __init__(self, config: Config):
        self.config = config
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Check device availability
        self.gpu_available = torch.cuda.is_available()
        if config.test_gpu and not self.gpu_available:
            print("Warning: GPU testing requested but CUDA not available")
        
        # Results storage
        self.results = []
        
    def create_random_input(self, seq_len: int, device: str) -> torch.Tensor:
        """
        Create random input tensor simulating sentence embeddings
        
        Args:
            seq_len: sequence length
            device: 'cpu' or 'cuda'
            
        Returns:
            Random tensor of shape (batch_size, seq_len, d_model)
        """
        input_tensor = torch.randn(
            self.config.batch_size, 
            seq_len, 
            self.config.d_model,
            device=device,
            dtype=torch.float32
        )
        return input_tensor
    
    def profile_attention_layer(self, model, input_tensor, attention_type: str, 
                              device: str, seq_len: int) -> Dict:
        """
        Profile a single attention layer
        
        Args:
            model: attention model (SingleHeadAttention or MultiHeadAttention)
            input_tensor: input tensor
            attention_type: 'single_head' or 'multi_head'
            device: 'cpu' or 'cuda'
            seq_len: sequence length
            
        Returns:
            Dictionary with profiling results
        """
        model.eval()
        
        # Count FLOPs
        flops = FLOPSCounter.count_flops(model, input_tensor)
        
        # Profile memory usage
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            with torch.no_grad():
                _ = model(input_tensor)
                
            torch.cuda.synchronize()
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            # For CPU, we'll measure process memory increase
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            with torch.no_grad():
                _ = model(input_tensor)
                
            final_memory = process.memory_info().rss
            peak_memory_mb = (final_memory - initial_memory) / (1024 * 1024)
            peak_memory_mb = max(peak_memory_mb, 0)  # Ensure non-negative
        
        # Time the forward pass
        def forward_pass():
            with torch.no_grad():
                return model(input_tensor)
        
        times = Timer.time_function(
            forward_pass, 
            num_runs=self.config.num_runs,
            warmup_runs=5,
            device=device
        )
        
        time_stats = compute_statistics(times)
        
        return {
            'seq_len': seq_len,
            'attention_type': attention_type,
            'device': device,
            'flops': flops,
            'memory_mb': peak_memory_mb,
            'times': times,
            'time_mean': time_stats['mean'],
            'time_std_error': time_stats['std_error']
        }
    
    def run_profiling_experiment(self):
        """
        Run the complete profiling experiment
        """
        print("Starting Self-Attention Profiling Experiment")
        print(f"Testing sequence lengths: {self.config.sequence_lengths}")
        print(f"Number of runs per measurement: {self.config.num_runs}")
        print(f"GPU available: {self.gpu_available}")
        print("-" * 50)
        
        # Test configurations
        attention_configs = []
        if self.config.test_single_head:
            attention_configs.append(('single_head', SingleHeadAttention))
        if self.config.test_multi_head:
            attention_configs.append(('multi_head', MultiHeadAttention))
        
        devices = []
        if self.config.test_cpu:
            devices.append('cpu')
        if self.config.test_gpu and self.gpu_available:
            devices.append('cuda')
        
        total_experiments = len(self.config.sequence_lengths) * len(attention_configs) * len(devices)
        pbar = tqdm(total=total_experiments, desc="Running experiments")
        
        # Run experiments
        for seq_len in self.config.sequence_lengths:
            for attention_type, attention_class in attention_configs:
                for device in devices:
                    try:
                        # Create model
                        if attention_type == 'single_head':
                            model = attention_class(self.config.d_model).to(device)
                        else:  # multi_head
                            model = attention_class(self.config.d_model, self.config.num_heads).to(device)
                        
                        # Create input
                        input_tensor = self.create_random_input(seq_len, device)
                        
                        # Profile
                        result = self.profile_attention_layer(
                            model, input_tensor, attention_type, device, seq_len
                        )
                        
                        self.results.append(result)
                        
                        pbar.set_postfix({
                            'seq_len': seq_len,
                            'attention': attention_type,
                            'device': device,
                            'time_ms': f"{result['time_mean']*1000:.2f}"
                        })
                        
                    except Exception as e:
                        print(f"Error in {attention_type} on {device} with seq_len {seq_len}: {e}")
                        continue
                    
                    pbar.update(1)
        
        pbar.close()
        print(f"\nCompleted {len(self.results)} experiments")
    
    def process_results(self) -> pd.DataFrame:
        """
        Process results into a pandas DataFrame with statistics
        """
        processed_results = []
        
        for result in self.results:
            # Compute FLOPS statistics (FLOPS is deterministic, so no error bars needed)
            flops_per_run = [result['flops']] * len(result['times'])
            flops_stats = compute_statistics(flops_per_run)
            
            # Memory is also typically deterministic for a given input
            memory_per_run = [result['memory_mb']] * len(result['times'])
            memory_stats = compute_statistics(memory_per_run)
            
            processed_results.append({
                'seq_len': result['seq_len'],
                'attention_type': result['attention_type'],
                'device': result['device'],
                'flops_mean': result['flops'],
                'flops_std_error': 0,  # FLOPS is deterministic
                'memory_mean': result['memory_mb'],
                'memory_std_error': 0,  # Memory usage is typically deterministic
                'time_mean': result['time_mean'],
                'time_std_error': result['time_std_error']
            })
        
        return pd.DataFrame(processed_results)
    
    def save_results(self, df: pd.DataFrame):
        """
        Save results to CSV files
        """
        if self.config.save_results:
            results_dir = Path(self.config.results_dir)
            results_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            df.to_csv(results_dir / "detailed_results.csv", index=False)
            
            # Save raw timing data
            raw_data = []
            for result in self.results:
                for i, time_val in enumerate(result['times']):
                    raw_data.append({
                        'seq_len': result['seq_len'],
                        'attention_type': result['attention_type'],
                        'device': result['device'],
                        'run': i,
                        'time': time_val,
                        'flops': result['flops'],
                        'memory_mb': result['memory_mb']
                    })
            
            raw_df = pd.DataFrame(raw_data)
            raw_df.to_csv(results_dir / "raw_timing_data.csv", index=False)
            
            print(f"Results saved to {results_dir}/")
    
    def generate_plots(self, df: pd.DataFrame):
        """
        Generate all plots
        """
        if self.config.save_plots:
            plotter = ResultsPlotter(self.config.results_dir)
            
            # Main complexity trends plot
            plotter.plot_complexity_trends(df, save=True)
            
            # Detailed scaling analysis
            plotter.plot_scaling_analysis(df, save=True)
            
            # Complexity comparison with theory
            plotter.plot_complexity_comparison(df, save=True)
            
            # Summary tables
            plotter.create_comparison_table(df, save=True)
            
            print(f"Plots saved to {self.config.results_dir}/")
    
    def print_summary(self, df: pd.DataFrame):
        """
        Print a summary of results
        """
        print("\n" + "="*60)
        print("PROFILING RESULTS SUMMARY")
        print("="*60)
        
        for attention_type in df['attention_type'].unique():
            print(f"\n{attention_type.upper().replace('_', '-')} ATTENTION:")
            print("-" * 30)
            
            subset = df[df['attention_type'] == attention_type]
            
            for device in subset['device'].unique():
                device_data = subset[subset['device'] == device]
                print(f"\n{device.upper()}:")
                
                for _, row in device_data.iterrows():
                    print(f"  Seq Len {row['seq_len']:>5d}: "
                          f"FLOPs={row['flops_mean']:>12.0f}, "
                          f"Memory={row['memory_mean']:>6.1f}MB, "
                          f"Time={row['time_mean']*1000:>8.2f}±{row['time_std_error']*1000:.2f}ms")


def main():
    """
    Main function to run the profiling experiment
    """
    config = Config()
    profiler = SelfAttentionProfiler(config)
    
    # Run the experiment
    profiler.run_profiling_experiment()
    
    # Process and analyze results
    results_df = profiler.process_results()
    
    # Save results
    profiler.save_results(results_df)
    
    # Generate plots
    profiler.generate_plots(results_df)
    
    # Print summary
    profiler.print_summary(results_df)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()