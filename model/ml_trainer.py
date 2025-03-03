import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import  Dict, List
from model.training_history import TrainingHistory
from model.parameter_tuner import ParameterTuner
from model.model_persistence import ModelPersistence
import pandas as pd
import os
            
class MLTrainer:
    def __init__(self):
        self.history = TrainingHistory()
        self.tuner = ParameterTuner()
        self.persistence = ModelPersistence()
        self.feature_scaler = StandardScaler()
        self.initialize_model()

    def initialize_model(self):
        """Initialize model with parameters and load previous state if exists"""
        # Try to load previous model state
        prev_model, prev_scaler = self.persistence.load_model()

        if prev_model is not None:
            self.model = prev_model
            if prev_scaler is not None:
                self.feature_scaler = prev_scaler
            print("Loaded previous model state")
        else:
            # Initialize new model
            params = self.tuner.current_params
            self.model = GradientBoostingRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                learning_rate=params['learning_rate'],
                subsample=0.8,
                loss='squared_error',
                random_state=42
            )
            # Use RobustScaler which is less sensitive to outliers
            self.feature_scaler = RobustScaler()

    def train(self, X: np.ndarray, y: np.ndarray, private_keys=None) -> Dict[str, float]:
        """Train the model and return metrics"""
        # Ensure numeric types
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        # Split data into train/test
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Track private keys if provided
        if private_keys is not None:
            # Convert private_keys to a list if it's not already
            if not isinstance(private_keys, list):
                private_keys = list(private_keys)
            keys_train = private_keys[:train_size]
            keys_test = private_keys[train_size:]
        else:
            keys_test = None

        # Load previous best candidates to guide training
        best_candidates = self.persistence.get_best_candidates_ever()
        if best_candidates:
            # Use previous best candidates to augment training data
            best_features = np.array([c['features'] for c in best_candidates])
            X_train = np.vstack([X_train, best_features])
            # Corresponding y values are already normalized
            y_train = np.concatenate([y_train, [c['X_Diff'] for c in best_candidates]])

        # Normalize features
        X_train = self.feature_scaler.fit_transform(X_train)
        X_test = self.feature_scaler.transform(X_test)

        # Train model
        self.model.fit(X_train, y_train)

        # Calculate metrics
        y_pred = self.model.predict(X_test)
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'min_x_diff': float(np.min(np.exp(y_test))),  # Convert back from log scale
            'feature_importance': self.model.feature_importances_.tolist()
        }

        # Save training history and model state
        best_candidates = self.get_best_candidates(X_test, y_test, keys_test)
        self.history.save_training_metrics(
            metrics=metrics,
            params=self.tuner.current_params,
            best_candidates=best_candidates
        )
        self.persistence.save_model(
            self.model, 
            self.feature_scaler, 
            metrics, 
            best_candidates
        )

        # Save the new generated keys to a CSV file
        if keys_test is not None:

            # Get indices of top 20 predictions
            top_indices = np.argsort(y_test)[:20]
            
            # Create DataFrame with private keys and their predicted X differences
            results_df = pd.DataFrame({
                'Private_Key': [f"0x{keys_test[i]:064x}" for i in top_indices],
                'X_Diff': [float(np.exp(y_test[i])) for i in top_indices]
            })
            
            # Ensure directory exists
            os.makedirs("generated_keys", exist_ok=True)
            
            # Save to CSV
            results_df.to_csv("generated_keys/predicted_keys.csv", index=False)
            print(f"Saved {len(top_indices)} new generated keys to generated_keys/predicted_keys.csv")

        # Update parameter tuner
        self.tuner.update_best_params(self.tuner.current_params, metrics['min_x_diff'])

        return metrics

    def get_best_candidates(self, X: np.ndarray, y: np.ndarray, private_keys=None, top_n: int = 10) -> List[Dict]:
        """Get the best candidates from the current batch"""
        indices = np.argsort(y)[:top_n]
        
        result = []
        for rank, i in enumerate(indices, 1):
            # Convert index to int to avoid slicing issues
            idx = int(i)
            
            candidate = {
                'features': X[idx].tolist(),
                'X_Diff': float(np.exp(y[idx])),  # Convert back from log scale
                'rank': int(rank)
            }
            
            # Add private key if available
            if private_keys is not None:
                candidate['private_key'] = private_keys[idx]
                
            result.append(candidate)
            
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model"""
        X = X.astype(np.float64)
        X = self.feature_scaler.transform(X)
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        feature_names = [
            'MSB', 'Bit_77', 'Bit_80', 'Bit_120',  # Binary features
            'Swapped_MSB', 'Swapped_Bit_77', 'Bit_Stability',  # Bit swap features
            'Hex_0', 'Hex_19', 'Hex_20', 'Hex_26',  # Hex features
            'Mod_40', 'Mod_7', 'Mod_256', 'Swapped_Mod_40', 'Swapped_Mod_7',  # Modular features
            'Poly_2', 'Poly_3', 'Poly_4'  # Polynomial terms
        ]

        # GradientBoostingRegressor uses different attribute for feature importance
        # Ensure the number of features matches the model's feature importances
        if len(feature_names) > len(self.model.feature_importances_):
            feature_names = feature_names[:len(self.model.feature_importances_)]
        elif len(feature_names) < len(self.model.feature_importances_):
            # Add generic names for any extra features
            for i in range(len(feature_names), len(self.model.feature_importances_)):
                feature_names.append(f'Feature_{i}')
        
        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def update_parameters(self):
        """Update model parameters based on historical performance"""
        history_metrics = self.history.get_progress_metrics().to_dict('records')
        new_params = self.tuner.suggest_parameters(history_metrics)
        self.tuner.current_params = new_params
        # Don't reinitialize model here - we want to keep the trained state
        # Just update parameters for next training session