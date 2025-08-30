import torch


class OptimizationFunction:
    """Base class for 2D optimization functions."""
    
    def __call__(self, x, y):
        """Compute function value at (x, y)."""
        raise NotImplementedError
    
    def get_name(self):
        """Return function name for plotting."""
        raise NotImplementedError


class QuadraticMinimization(OptimizationFunction):
    """Quadratic function f(x,y) = x² + y² for minimization."""
    
    def __call__(self, x, y):
        """Compute f(x,y) = x² + y²."""
        return x**2 + y**2
    
    def get_name(self):
        """Return function name."""
        return "f(x,y) = x² + y²"


class QuadraticMaximization(OptimizationFunction):
    """Negative quadratic function f(x,y) = -x² - y² for maximization."""
    
    def __call__(self, x, y):
        """Compute f(x,y) = -x² - y²."""
        return -(x**2 + y**2)
    
    def get_name(self):
        """Return function name."""
        return "f(x,y) = -x² - y²"