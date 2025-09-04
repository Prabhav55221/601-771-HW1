"""Configuration for SGD optimization experiments.

Author: Prabhav Singh
"""


class Config:
    """Configuration parameters for optimization experiments."""
    learning_rate = 0.1
    num_iterations = 100
    starting_x = 2.0
    starting_y = 2.0
    
    momentum_values = [0.0, 0.3, 0.6, 0.9]
    weight_decay = 0.1
    
    plot_range = 3.5
    contour_levels = 20
    figure_size = (12, 4)
    
    random_seed = 42