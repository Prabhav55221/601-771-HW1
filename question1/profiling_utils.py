"""
Utilities for profiling self-attention: FLOPS counting, memory usage, and timing
"""

import torch
import time
import psutil
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple

try:
    from torchprofile import profile_macs
    TORCHPROFILE_AVAILABLE = True
except ImportError:
    TORCHPROFILE_AVAILABLE = False

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False


class FLOPSCounter:
    """
    Count floating point operations using available libraries
    """
    
    @staticmethod
    def count_flops(model, input_tensor):
        """
        Count FLOPs for a single forward pass
        
        Args:
            model: PyTorch model
            input_tensor: input tensor
            
        Returns:
            int: number of floating point operations
        """
        if FVCORE_AVAILABLE:
            return FLOPSCounter._count_with_fvcore(model, input_tensor)
        elif TORCHPROFILE_AVAILABLE:
            return FLOPSCounter._count_with_torchprofile(model, input_tensor)
        else:
            return FLOPSCounter._count_manual(model, input_tensor)
    
    @staticmethod
    def _count_with_fvcore(model, input_tensor):
        """Use fvcore library for FLOP counting"""
        model.eval()
        with torch.no_grad():
            flop_dict, _ = flop_count(model, (input_tensor,), supported_ops=None)
            return sum(flop_dict.values())
    
    @staticmethod
    def _count_with_torchprofile(model, input_tensor):
        """Use torchprofile library for MAC counting (multiply-accumulate operations)"""
        model.eval()
        with torch.no_grad():
            macs = profile_macs(model, input_tensor)
            # Each MAC involves 2 FLOPs (multiply + accumulate)
            return macs * 2
    
    @staticmethod
    def _count_manual(model, input_tensor):
        """Manual FLOP counting for attention operations"""
        batch_size, seq_len, d_model = input_tensor.shape
        
        if hasattr(model, 'num_heads'):  # Multi-head attention
            num_heads = model.num_heads
            d_k = d_model // num_heads
            
            # Q, K, V projections: 3 * (seq_len * d_model * d_model)
            qkv_flops = 3 * seq_len * d_model * d_model
            
            # Attention computation per head: seq_len^2 * d_k for QK^T, seq_len^2 * d_k for attention * V
            attention_flops = num_heads * (2 * seq_len * seq_len * d_k)
            
            # Output projection: seq_len * d_model * d_model
            output_flops = seq_len * d_model * d_model
            
            total_flops = qkv_flops + attention_flops + output_flops
        else:  # Single-head attention
            # Q, K, V projections: 3 * (seq_len * d_model * d_model)
            qkv_flops = 3 * seq_len * d_model * d_model
            
            # Attention computation: seq_len^2 * d_model for QK^T, seq_len^2 * d_model for attention * V
            attention_flops = 2 * seq_len * seq_len * d_model
            
            total_flops = qkv_flops + attention_flops
        
        return total_flops * batch_size


class MemoryProfiler:
    """
    Profile memory usage for GPU and CPU
    """
    
    @staticmethod
    @contextmanager
    def profile_gpu_memory():
        """Context manager to profile GPU memory usage"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            yield
            
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            yield
            peak_memory = 0
            
        yield peak_memory
    
    @staticmethod
    @contextmanager
    def profile_cpu_memory():
        """Context manager to profile CPU memory usage"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        yield
        
        peak_memory = process.memory_info().rss - initial_memory
        yield peak_memory


class Timer:
    """
    High-precision timing utilities
    """
    
    @staticmethod
    def time_function(func, *args, num_runs=1, warmup_runs=3, device='cpu'):
        """
        Time a function with multiple runs and warmup
        
        Args:
            func: function to time
            args: arguments to pass to function
            num_runs: number of timing runs
            warmup_runs: number of warmup runs (not counted)
            device: 'cpu' or 'cuda'
            
        Returns:
            List[float]: list of timing results in seconds
        """
        # Warmup runs
        for _ in range(warmup_runs):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = func(*args)
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Actual timing runs
        times = []
        for _ in range(num_runs):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            _ = func(*args)
            
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return times


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute mean and standard error for a list of values
    
    Args:
        values: list of numerical values
        
    Returns:
        dict with 'mean' and 'std_error' keys
    """
    import numpy as np
    
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)  # sample standard deviation
    std_error = std_dev / np.sqrt(len(values))  # standard error
    
    return {
        'mean': mean,
        'std_error': std_error,
        'std_dev': std_dev,
        'count': len(values)
    }