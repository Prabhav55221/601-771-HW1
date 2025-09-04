# Question 3: SGD Optimization Analysis

PyTorch SGD optimization experiments on 2D quadratic functions.

## Overview

Analyzes how SGD hyperparameters affect optimization trajectories:
- **Momentum effects**: Varying momentum from 0 to 0.9
- **Weight decay impact**: L2 regularization effects  
- **Maximization behavior**: Using maximize=True flag

## Usage

### SLURM Execution
```bash
sbatch run_experiment.sh
```

### Local Testing
```bash
python optimizer_experiment.py
```

## Experiments

- **(a)** Momentum comparison on f(x,y) = x² + y²
- **(b)** Weight decay effects with momentum
- **(c)** Maximization on f(x,y) = -x² - y²

## Results

- Contour plots with optimization trajectories
- Numerical analysis of final convergence points
- Visual comparison of hyperparameter effects