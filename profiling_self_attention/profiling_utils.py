"""Profiling utilities for self-attention analysis.

Author: Prabhav Singh
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
    """FLOPS counting utilities using PyTorch profiler and fallback methods."""
    
    @staticmethod
    def count_flops(model, input_tensor):
        """Count FLOPs for model inference."""
        return FLOPSCounter._count_with_pytorch_profiler(model, input_tensor)
    
    @staticmethod
    def _count_with_pytorch_profiler(model, input_tensor):
        """Count FLOPs using PyTorch profiler."""
        model.eval()
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_flops=True
            ) as prof:
                _ = model(input_tensor)
            
            total_flops = sum([event.flops for event in prof.events() if event.flops > 0])
            return total_flops if total_flops > 0 else FLOPSCounter._count_with_fallback(model, input_tensor)
    
    @staticmethod
    def _count_with_fallback(model, input_tensor):
        """Fallback FLOPS counting methods."""
        if FVCORE_AVAILABLE:
            return FLOPSCounter._count_with_fvcore(model, input_tensor)
        elif TORCHPROFILE_AVAILABLE:
            return FLOPSCounter._count_with_torchprofile(model, input_tensor)
        else:
            raise RuntimeError("No FLOPS counting library available. Please install fvcore or torchprofile.")
    
    @staticmethod
    def _count_with_fvcore(model, input_tensor):
        """Count FLOPs using fvcore."""
        model.eval()
        with torch.no_grad():
            flop_dict, _ = flop_count(model, (input_tensor,), supported_ops=None)
            return sum(flop_dict.values())
    
    @staticmethod
    def _count_with_torchprofile(model, input_tensor):
        """Count FLOPs using torchprofile."""
        model.eval()
        with torch.no_grad():
            macs = profile_macs(model, input_tensor)
            return macs * 2


class Timer:
    """Timing utilities for performance measurement."""
    
    @staticmethod
    def time_function(func, *args, num_runs=1, warmup_runs=3, device='cpu'):
        """Time function execution with warmup and multiple runs."""
        for _ in range(warmup_runs):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = func(*args)
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        times = []
        for run_idx in range(num_runs):
            torch.manual_seed(torch.randint(0, 10000, (1,)).item())
            
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
    """Compute statistical measures for timing data."""
    import numpy as np
    
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    std_error = std_dev / np.sqrt(len(values))
    
    return {
        'mean': mean,
        'std_error': std_error,
        'std_dev': std_dev,
        'count': len(values)
    }