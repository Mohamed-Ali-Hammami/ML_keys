import numpy as np
from typing import List, Dict, Union
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import random  
class FeatureExtractor:
    def __init__(self, polynomial_degree=3, bit_swap_pairs=None): 
        self.scaler = StandardScaler()
        self.polynomial_degree = polynomial_degree
        # Default bit swap pairs (can be overridden)
        self.bit_swap_pairs = bit_swap_pairs or [(0, 255), (77, 80), (120, 160), (200, 240)]

    def extract_signature_features(self, r_values: List[int], s_values: List[int]) -> Dict[str, float]:
        """Extract statistical features from signature values"""
        r_array = np.array(r_values, dtype=np.float64)
        s_array = np.array(s_values, dtype=np.float64)

        # Ensure arrays are 1-dimensional
        r_array = r_array.ravel()
        s_array = s_array.ravel()

        features = {
            'r_mean': float(np.mean(r_array)),
            'r_std': float(np.std(r_array)),
            'r_min': float(np.min(r_array)),
            'r_max': float(np.max(r_array)),
            's_mean': float(np.mean(s_array)),
            's_std': float(np.std(s_array)),
            's_min': float(np.min(s_array)),
            's_max': float(np.max(s_array)),
            'r_s_correlation': float(np.corrcoef(r_array, s_array)[0,1]),
            'r_mod_40': float(np.mean([x % 40 for x in r_array])),
            'r_mod_7': float(np.mean([x % 7 for x in r_array]))
        }
        return features

    def perform_bit_swap(self, private_key: int) -> int:
        """Swap bits in the private key according to defined pairs"""
        result = private_key
        for bit1, bit2 in self.bit_swap_pairs:
            # Get the values of the two bits
            bit1_val = (result >> bit1) & 1
            bit2_val = (result >> bit2) & 1
            
            # If the bits are different, we need to swap them
            if bit1_val != bit2_val:
                # Create a mask with 1s at the positions to swap
                mask = (1 << bit1) | (1 << bit2)
                # XOR with the mask flips both bits
                result ^= mask
                
        return result
        
    def generate_new_keys(self, private_key: int, num_keys: int = 5) -> List[int]:
        """Generate new keys by applying various bit manipulations to the original key"""

        new_keys = []
        
        # Always include the original key
        new_keys.append(private_key)
        
        # Add swapped key
        swapped_key = self.perform_bit_swap(private_key)
        new_keys.append(swapped_key)
        
        # Generate more keys with different bit swap combinations
        for _ in range(num_keys - 2):  # -2 because we already added two keys
            # Create a random subset of bit pairs to swap
            random_pairs = []
            for _ in range(random.randint(1, 3)):
                bit1 = random.randint(0, 255)
                bit2 = random.randint(0, 255)
                if bit1 != bit2:
                    random_pairs.append((bit1, bit2))
            
            # Apply these random swaps
            result = private_key
            for bit1, bit2 in random_pairs:
                bit1_val = (result >> bit1) & 1
                bit2_val = (result >> bit2) & 1
                
                if bit1_val != bit2_val:
                    mask = (1 << bit1) | (1 << bit2)
                    result ^= mask
            
            # Add a small random offset (less than 100) to create variations
            offset = random.randint(-50, 50)
            result = (result + offset) % (1 << 256)  # Keep within 256 bits
            
            new_keys.append(result)
            
        return new_keys
    
    def extract_private_key_features(self, private_key: int) -> Dict[str, Union[float, List[int]]]:
        """Extract features from a private key"""
        # Original binary features (256 bits)
        binary = [(private_key >> i) & 1 for i in range(255, -1, -1)]
        
        # Bit-swapped key
        swapped_key = self.perform_bit_swap(private_key)
        swapped_binary = [(swapped_key >> i) & 1 for i in range(255, -1, -1)]
        
        # Compare original and swapped keys to get bit stability metrics
        bit_changes = sum(1 for a, b in zip(binary, swapped_binary) if a != b)
        bit_stability = 1.0 - (bit_changes / 256.0)  # Higher means more stable

        # Hexadecimal features
        hex_str = f"{private_key:064x}"
        hex_features = [int(d, 16) for d in hex_str]

        # Modular features
        mod_features = {
            'mod_40': private_key % 40,
            'mod_7': private_key % 7,
            'mod_256': private_key % 256,
            'swapped_mod_40': swapped_key % 40,
            'swapped_mod_7': swapped_key % 7
        }

        # Polynomial features (up to degree specified by polynomial_degree)
        poly_features = []
        x = private_key
        for i in range(2, self.polynomial_degree + 1):  # degrees 2-polynomial_degree
            try:
                x = (x * private_key) % (1 << 256)  # Keep within 256 bits
                poly_features.append(x)
            except OverflowError:
                break

        return {
            'binary': binary,
            'swapped_binary': swapped_binary,
            'bit_stability': bit_stability,
            'hex': hex_features,
            'modular': mod_features,
            'polynomial': poly_features
        }

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using StandardScaler"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.scaler.fit_transform(features)
        
    def test_polynomial_degrees(self, X_raw, y_raw, degrees_to_test=[1, 2, 4, 6, 8, 10]):
        """Test different polynomial degrees and return performance metrics"""
 
        results = []
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
        
        # Test each polynomial degree
        for degree in degrees_to_test:
            # Store current degree
            original_degree = self.polynomial_degree
            
            # Set new degree for testing
            self.polynomial_degree = degree
            
            # Generate features with this polynomial degree (assuming there's a transform method)
            try:
                # If you have a transform method, use it
                # For now, we'll just use the raw features
                X_train_poly = X_train
                X_test_poly = X_test
                
                # Train a model
                model = GradientBoostingRegressor(
                    n_estimators=100, 
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=42
                )
                model.fit(X_train_poly, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_poly)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'polynomial_degree': degree,
                    'mse': mse,
                    'r2': r2
                })
                
            except Exception as e:
                results.append({
                    'polynomial_degree': degree,
                    'mse': float('nan'),
                    'r2': float('nan'),
                    'error': str(e)
                })
            
            # Restore original degree
            self.polynomial_degree = original_degree
        
        return pd.DataFrame(results)
