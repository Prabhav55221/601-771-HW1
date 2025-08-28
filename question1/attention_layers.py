"""
Self-attention layer implementations: single-head and multi-head
Following "Attention is All You Need" paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    """
    Single-head scaled dot-product attention
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.sqrt_d_model = math.sqrt(d_model)
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: attention output of shape (batch_size, seq_len, d_model)
        """
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Scaled dot-product attention
        # scores = QK^T / sqrt(d_model)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d_model
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as described in "Attention is All You Need"
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sqrt_d_k = math.sqrt(self.d_k)
        
        # Linear projections for Q, K, V (for all heads)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: attention output of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention for each head
        # Q, K, V are now (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d_k
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(attention_output)
        
        return output