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
    @staticmethod
    def count_flops(model, input_tensor):
        if FVCORE_AVAILABLE:
            return FLOPSCounter._count_with_fvcore(model, input_tensor)
        elif TORCHPROFILE_AVAILABLE:
            return FLOPSCounter._count_with_torchprofile(model, input_tensor)
        else:
            return FLOPSCounter._count_manual(model, input_tensor)
    
    @staticmethod
    def _count_with_fvcore(model, input_tensor):
        model.eval()
        with torch.no_grad():
            flop_dict, _ = flop_count(model, (input_tensor,), supported_ops=None)
            return sum(flop_dict.values())
    
    @staticmethod
    def _count_with_torchprofile(model, input_tensor):
        model.eval()
        with torch.no_grad():
            macs = profile_macs(model, input_tensor)
            return macs * 2
    
    @staticmethod
    def _count_manual(model, input_tensor):
        batch_size, seq_len, d_model = input_tensor.shape
        
        if hasattr(model, 'num_heads'):
            num_heads = model.num_heads
            d_k = d_model // num_heads
            qkv_flops = 3 * seq_len * d_model * d_model
            attention_flops = num_heads * (2 * seq_len * seq_len * d_k)
            output_flops = seq_len * d_model * d_model
            total_flops = qkv_flops + attention_flops + output_flops
        else:
            qkv_flops = 3 * seq_len * d_model * d_model
            attention_flops = 2 * seq_len * seq_len * d_model
            total_flops = qkv_flops + attention_flops
        
        return total_flops * batch_size


class Timer:
    @staticmethod
    def time_function(func, *args, num_runs=1, warmup_runs=3, device='cpu'):
        for _ in range(warmup_runs):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = func(*args)
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
        
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