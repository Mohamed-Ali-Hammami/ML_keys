from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import ParameterGrid

class ParameterTuner:
    def __init__(self):
        self.param_ranges = {
            'n_estimators': [100, 150, 200, 250],
            'max_depth': [3, 4, 5, 6],  # Reduced depth to prevent overfitting
            'min_samples_leaf': [5, 7, 9, 11],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate for gradient boosting
            'polynomial_degree': [2, 3, 4],  # Reduced polynomial degree
            'feature_selection_threshold': [0.03, 0.05, 0.07]
        }
        
        self.current_params = {
            'n_estimators': 150,
            'max_depth': 4,  # Lower max_depth to reduce overfitting
            'min_samples_leaf': 9,  # Increased to reduce overfitting
            'learning_rate': 0.05,  # New parameter for gradient boosting
            'polynomial_degree': 3,  # Reduced to prevent overfitting
            'feature_selection_threshold': 0.05
        }
        
        self.best_params = self.current_params.copy()
        self.best_score = float('inf')
        
    def suggest_parameters(self, history_metrics: List[Dict]) -> Dict:
        """Suggest next set of parameters based on historical performance"""
        if not history_metrics:
            return self.current_params
            
        # Calculate performance trends
        x_diffs = [m['min_x_diff'] for m in history_metrics]
        improvements = np.diff(x_diffs)
        
        # If we're improving, stick with current direction
        if len(improvements) > 0 and np.mean(improvements) < 0:
            return self.current_params
            
        # Otherwise, explore new parameter combinations
        param_grid = list(ParameterGrid(self.param_ranges))
        next_params = np.random.choice(param_grid)
        
        return {**self.current_params, **next_params}
        
    def update_best_params(self, params: Dict, score: float):
        """Update best parameters if new score is better"""
        if score < self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.current_params = params.copy()
            
    def get_exploration_rate(self, session_count: int) -> float:
        """Calculate exploration rate that decreases over time"""
        return max(0.1, 1.0 / (1 + session_count * 0.1))
        
    def should_explore(self, session_count: int) -> bool:
        """Decide whether to explore new parameters or exploit current ones"""
        exploration_rate = self.get_exploration_rate(session_count)
        return np.random.random() < exploration_rate
