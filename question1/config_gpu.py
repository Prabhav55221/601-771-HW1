class Config:
    d_model = 768
    num_heads = 8
    batch_size = 1
    sequence_lengths = [10, 100, 1000, 10000]
    num_runs = 20
    random_seed = 42
    test_single_head = True
    test_multi_head = True
    test_cpu = False
    test_gpu = True