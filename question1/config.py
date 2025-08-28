"""
Configuration file for self-attention profiling experiment
"""

class Config:
    # Model hyperparameters
    d_model = 768          # embedding dimension (configurable)
    num_heads = 8          # number of attention heads for multi-head attention
    batch_size = 1         # batch size for profiling
    
    # Experiment settings
    sequence_lengths = [10, 100, 1000, 10000]  # input lengths to test
    num_runs = 20          # number of runs for averaging and error calculation
    random_seed = 42       # for reproducibility
    
    # Profiling options
    test_single_head = True    # whether to test single-head attention
    test_multi_head = True     # whether to test multi-head attention
    test_cpu = True           # whether to test on CPU
    test_gpu = True           # whether to test on GPU (if available)
    
    # Output settings
    save_results = True       # save numerical results to CSV
    save_plots = True         # save plots to files
    results_dir = "results/"  # directory to save results