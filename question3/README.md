# Question 3: PyTorch SGD Optimization Experiments

Optimization experiments comparing SGD hyperparameter effects on 2D functions.

## Files

- `config.py` - Configuration parameters for experiments
- `optimization_functions.py` - QuadraticMinimization and QuadraticMaximization classes
- `optimizer_experiment.py` - Main experiment orchestrator
- `plotting_utils.py` - Contour plot and trajectory visualization utilities

## Usage

1. Create conda environment:
```bash
conda env create -f ../environment.yml
conda activate nlp-hw1
```

2. Run experiments:
```bash
python optimizer_experiment.py
```

## Experiments

**(a) Momentum Comparison**: Tests momentum values [0, 0.3, 0.6, 0.9] on f(x,y) = x² + y²
**(b) Weight Decay Effects**: Compares optimization with/without weight_decay=0.1
**(c) Maximization**: Uses maximize=True on f(x,y) = -x² - y²

## Output

- `results/momentum_comparison.png` - Momentum trajectory comparison
- `results/weight_decay_comparison.png` - Weight decay effect visualization  
- `results/maximization_experiment.png` - Maximization behavior
- `results/optimization_analysis.txt` - Numerical results summary