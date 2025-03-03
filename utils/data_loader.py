import pandas as pd
import numpy as np
from typing import Tuple
from utils.ecdsa_lib import compute_public_key
from sklearn.preprocessing import StandardScaler
from model.feature_extractor import FeatureExtractor
from utils.ecdsa_lib import compute_public_key
class DataLoader:
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    @staticmethod
    def load_signature_data(file_path: str) -> pd.DataFrame:
        """Load signature data from CSV file"""
        df = pd.read_csv(file_path)
        # Convert hex strings to integers using Python's int
        df['r'] = df['r'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x))
        df['s'] = df['s'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x))
        return df

    @staticmethod
    def load_candidates(file_path: str) -> pd.DataFrame:
        """Load candidate private keys and x-differences"""
        df = pd.read_csv(file_path)
        # Convert hex strings to Python int and ensure numeric X_Diff
        df['Private_Key'] = df['Private_Key'].apply(lambda x: int(str(x), 16) if isinstance(x, str) else int(x))
        df['X_Diff'] = pd.to_numeric(df['X_Diff'], errors='coerce')
        return df

    def prepare_data(self, sig_data_40: pd.DataFrame, sig_data_7: pd.DataFrame, candidates: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training with generated keys"""
        features = []
        target_values = []
        target_x = 54466516474177380511287022884940819018437102136646648862302418152034269010364

        
        # Create feature extractor for bit swapping
        feature_extractor = FeatureExtractor(
            bit_swap_pairs=[(0, 255), (77, 80), (120, 160), (200, 240)]
        )

        # Process candidates and generate new keys
        expanded_keys = []
        expanded_x_diffs = []
        
        for _, candidate in candidates.iterrows():
            private_key = int(candidate['Private_Key'])  # Ensure Python int
            original_x_diff = float(candidate['X_Diff'])
            
            # Generate new keys based on this candidate
            new_keys = feature_extractor.generate_new_keys(private_key, num_keys=5)
            
            # Calculate X-coordinate differences for each new key
            for new_key in new_keys:
                try:
                    # Ensure new_key is an integer
                    new_key = int(new_key)
                    
                    # Compute public key X coordinate
                    pub_x = compute_public_key(new_key)[0]
                    
                    # Calculate difference with target X
                    x_diff = abs(pub_x - target_x)
                    
                    # Add to expanded lists
                    expanded_keys.append(new_key)
                    expanded_x_diffs.append(x_diff)
                except Exception as e:
                    # If key computation fails, use original values
                    expanded_keys.append(private_key)
                    expanded_x_diffs.append(original_x_diff)
        
        # Process each expanded key
        for i, private_key in enumerate(expanded_keys):
            # Perform bit swapping
            swapped_key = feature_extractor.perform_bit_swap(private_key)
            
            # Calculate bit stability
            binary_orig = [(private_key >> i) & 1 for i in range(255, -1, -1)]
            binary_swapped = [(swapped_key >> i) & 1 for i in range(255, -1, -1)]
            bit_changes = sum(1 for a, b in zip(binary_orig, binary_swapped) if a != b)
            bit_stability = 1.0 - (bit_changes / 256.0)  # Normalized stability metric

            # Binary features (significant bits identified in analysis)
            binary_features = [
                (private_key >> 255) & 1,         # MSB (Bit 0)
                (private_key >> (255-77)) & 1,    # Bit 77
                (private_key >> (255-80)) & 1,    # Bit 80
                (private_key >> (255-120)) & 1,   # Bit 120
                # Add swapped bit features
                (swapped_key >> 255) & 1,         # Swapped MSB
                (swapped_key >> (255-77)) & 1,    # Swapped Bit 77
                bit_stability                     # Bit stability metric
            ]

            # Hex digit features (positions with high correlation)
            hex_str = f"{private_key:064x}"
            hex_features = [
                int(hex_str[0], 16),     # Position 0 (MSB)
                int(hex_str[19], 16),    # Position 19
                int(hex_str[20], 16),    # Position 20
                int(hex_str[26], 16)     # Position 26
            ]

            # Modular features (from signature analysis)
            mod_features = [
                private_key % 40,
                private_key % 7,
                private_key % 256,
                swapped_key % 40,        # Swapped key modulo 40
                swapped_key % 7         # Swapped key modulo 7
            ]

            # Polynomial features (reduced to avoid overflow)
            poly_features = []
            x = private_key % (1 << 64)  # Use only lower 64 bits
            for degree in range(2, 5):  # degrees 2-4
                x = (x * (private_key % (1 << 32))) % (1 << 64)
                poly_features.append(float(x))  # Convert to float early

            # Combine all features
            feature_vector = (
                binary_features +
                hex_features +
                mod_features +
                poly_features
            )
            features.append(feature_vector)
            target_values.append(expanded_x_diffs[i])

        # Convert to numpy arrays with proper dtype
        X = np.array(features, dtype=np.float64)

        # Transform target values using log scale to handle large differences
        y = np.array(target_values, dtype=np.float64)
        y = np.log1p(y)  # log1p(x) = log(1 + x) to handle zeros

        # Normalize both features and target
        X = self.feature_scaler.fit_transform(X)
        y = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Print some statistics about the expanded dataset
        print(f"Generated expanded dataset with {len(X)} entries (original: {len(candidates)})")
        print(f"Best X_Diff in expanded set: {min(expanded_x_diffs):.2e}")

        return X, y, expanded_keys

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        """Convert predictions back to original scale"""
        y_pred = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return np.expm1(y_pred)  # inverse of log1p

    @staticmethod
    def validate_key(private_key: int, target_x: int) -> bool:
        """Validate if a private key generates the target x-coordinate"""
        try:
            pub_x = compute_public_key(private_key)[0]
            return pub_x == target_x
        except Exception:
            return False