"""Self-attention profiling and analysis.

Author: Prabhav Singh
"""
import torch
import pandas as pd
import numpy as np
from typing import Dict, List
import psutil
import os
from tqdm import tqdm

from config import Config
from attention_layers import SingleHeadAttention, MultiHeadAttention
from profiling_utils import FLOPSCounter, Timer, compute_statistics
from plot_results import plot_complexity_trends, plot_scaling_analysis


class SelfAttentionProfiler:
    """Profiles self-attention mechanisms for computational analysis."""
    
    def __init__(self, config: Config):
        """Initialize profiler with configuration."""
        self.config = config
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        self.gpu_available = torch.cuda.is_available()
        if config.test_gpu and not self.gpu_available:
            print("Warning: GPU testing requested but CUDA not available")
        
        self.results = []
        
    def create_random_input(self, seq_len: int, device: str) -> torch.Tensor:
        """Create random input tensor for profiling."""
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
        """Profile attention layer for FLOPs, memory, and timing."""
        model.eval()
        
        flops = FLOPSCounter.count_flops(model, input_tensor)
        
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            with torch.no_grad():
                _ = model(input_tensor)
                
            torch.cuda.synchronize()
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            batch_size, seq_len, d_model = input_tensor.shape
            
            input_mem = input_tensor.numel() * 4
            qkv_mem = 3 * input_mem
            attention_mem = batch_size * seq_len * seq_len * 4
            
            if hasattr(model, 'num_heads'):
                attention_mem = batch_size * model.num_heads * seq_len * seq_len * 4
            
            total_mem_bytes = input_mem + qkv_mem + attention_mem
            peak_memory_mb = total_mem_bytes / (1024 * 1024)
        
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
        """Run complete profiling experiment across configurations."""
        print("Starting Self-Attention Profiling Experiment")
        print(f"Testing sequence lengths: {self.config.sequence_lengths}")
        print(f"Number of runs per measurement: {self.config.num_runs}")
        print(f"GPU available: {self.gpu_available}")
        print("-" * 50)
        
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
        
        for seq_len in self.config.sequence_lengths:
            for attention_type, attention_class in attention_configs:
                for device in devices:
                    try:
                        if attention_type == 'single_head':
                            model = attention_class(self.config.d_model).to(device)
                        else:
                            model = attention_class(self.config.d_model, self.config.num_heads).to(device)
                        
                        input_tensor = self.create_random_input(seq_len, device)
                        
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
        """Process raw results into structured DataFrame."""
        processed_results = []
        
        for result in self.results:
            processed_results.append({
                'seq_len': result['seq_len'],
                'attention_type': result['attention_type'],
                'device': result['device'],
                'flops_mean': result['flops'],
                'flops_std_error': 0,
                'memory_mean': result['memory_mb'],
                'memory_std_error': 0,
                'time_mean': result['time_mean'],
                'time_std_error': result['time_std_error']
            })
        
        return pd.DataFrame(processed_results)
    
    def generate_plots(self, df: pd.DataFrame):
        """Generate complexity and scaling plots."""
        plot_complexity_trends(df)
        plot_scaling_analysis(df)
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary of profiling results."""
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
    config = Config()
    profiler = SelfAttentionProfiler(config)
    
    profiler.run_profiling_experiment()
    results_df = profiler.process_results()
    profiler.generate_plots(results_df)
    profiler.print_summary(results_df)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()