import torch
import numpy as np
from typing import List, Dict, Tuple
import os

from config import Config
from optimization_functions import QuadraticMinimization, QuadraticMaximization
from plotting_utils import (save_momentum_comparison_plot, 
                           save_weight_decay_comparison_plot,
                           save_maximization_plot)


class OptimizerExperiment:
    """Orchestrates PyTorch optimization experiments with SGD."""
    
    def __init__(self, config: Config):
        self.config = config
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def run_optimization(self, func, momentum: float = 0.0, weight_decay: float = 0.0, 
                        maximize: bool = False) -> List[Tuple[float, float]]:
        """Run single optimization experiment and return trajectory."""
        x = torch.tensor(self.config.starting_x, requires_grad=True, dtype=torch.float32)
        y = torch.tensor(self.config.starting_y, requires_grad=True, dtype=torch.float32)
        
        optimizer = torch.optim.SGD([x, y], lr=self.config.learning_rate,
                                   momentum=momentum, weight_decay=weight_decay,
                                   maximize=maximize)
        
        trajectory = []
        
        for iteration in range(self.config.num_iterations):
            optimizer.zero_grad()
            
            loss = func(x, y)
            trajectory.append((x.item(), y.item()))
            
            loss.backward()
            optimizer.step()
        
        return trajectory
    
    def experiment_a_momentum_comparison(self):
        """Experiment (a): Compare different momentum values."""
        func = QuadraticMinimization()
        trajectories = {}
        
        for momentum in self.config.momentum_values:
            trajectory = self.run_optimization(func, momentum=momentum)
            trajectories[momentum] = trajectory
        
        save_momentum_comparison_plot(func, trajectories, self.config, 
                                    'momentum_comparison.png')
        return trajectories
    
    def experiment_b_weight_decay_comparison(self):
        """Experiment (b): Compare with and without weight decay."""
        func = QuadraticMinimization()
        
        trajectories_no_decay = {}
        trajectories_with_decay = {}
        
        for momentum in self.config.momentum_values:
            trajectory_no_decay = self.run_optimization(func, momentum=momentum)
            trajectory_with_decay = self.run_optimization(func, momentum=momentum,
                                                        weight_decay=self.config.weight_decay)
            
            trajectories_no_decay[momentum] = trajectory_no_decay
            trajectories_with_decay[momentum] = trajectory_with_decay
        
        save_weight_decay_comparison_plot(func, trajectories_no_decay, 
                                        trajectories_with_decay, self.config,
                                        'weight_decay_comparison.png')
        
        return trajectories_no_decay, trajectories_with_decay
    
    def experiment_c_maximization(self):
        """Experiment (c): Test maximize=True on negative function."""
        func = QuadraticMaximization()
        trajectories = {}
        
        for momentum in self.config.momentum_values:
            trajectory = self.run_optimization(func, momentum=momentum, maximize=True)
            trajectories[f'momentum={momentum}'] = trajectory
        
        save_maximization_plot(func, trajectories, self.config,
                             'maximization_experiment.png')
        return trajectories
    
    def run_all_experiments(self):
        """Run all three experiments and save results."""
        os.makedirs('results', exist_ok=True)
        
        print("Running Experiment (a): Momentum Comparison")
        trajectories_a = self.experiment_a_momentum_comparison()
        
        print("Running Experiment (b): Weight Decay Comparison")
        trajectories_b_no_decay, trajectories_b_with_decay = self.experiment_b_weight_decay_comparison()
        
        print("Running Experiment (c): Maximization Experiment")
        trajectories_c = self.experiment_c_maximization()
        
        self.save_analysis_results(trajectories_a, trajectories_b_no_decay, 
                                 trajectories_b_with_decay, trajectories_c)
        
        print("All experiments completed. Results saved to results/ directory.")
    
    def save_analysis_results(self, trajectories_a, trajectories_b_no_decay,
                            trajectories_b_with_decay, trajectories_c):
        """Save numerical analysis of experiments."""
        with open('results/optimization_analysis.txt', 'w') as f:
            f.write("PyTorch SGD Optimization Experiments Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Experiment (a): Momentum Comparison on f(x,y) = x² + y²\n")
            f.write("-" * 50 + "\n")
            for momentum, trajectory in trajectories_a.items():
                final_point = trajectory[-1]
                f.write(f"Momentum {momentum}: Final point = ({final_point[0]:.4f}, {final_point[1]:.4f})\n")
            
            f.write(f"\nExperiment (b): Weight Decay Comparison\n")
            f.write("-" * 50 + "\n")
            f.write("Without Weight Decay:\n")
            for momentum, trajectory in trajectories_b_no_decay.items():
                final_point = trajectory[-1]
                f.write(f"  Momentum {momentum}: Final point = ({final_point[0]:.4f}, {final_point[1]:.4f})\n")
            
            f.write("With Weight Decay (0.1):\n")
            for momentum, trajectory in trajectories_b_with_decay.items():
                final_point = trajectory[-1]
                f.write(f"  Momentum {momentum}: Final point = ({final_point[0]:.4f}, {final_point[1]:.4f})\n")
            
            f.write(f"\nExperiment (c): Maximization with maximize=True on f(x,y) = -x² - y²\n")
            f.write("-" * 50 + "\n")
            for label, trajectory in trajectories_c.items():
                final_point = trajectory[-1]
                f.write(f"{label}: Final point = ({final_point[0]:.4f}, {final_point[1]:.4f})\n")


def main():
    """Main function to run all optimization experiments."""
    config = Config()
    experiment = OptimizerExperiment(config)
    experiment.run_all_experiments()


if __name__ == "__main__":
    main()