
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

class ModelPersistence:
    def __init__(self, base_dir: str = "model_checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.model_file = self.base_dir / "latest_model.pkl"
        self.scaler_file = self.base_dir / "feature_scaler.pkl"
        self.candidates_file = self.base_dir / "best_candidates.pkl"
        self.history_file = self.base_dir / "training_history.pkl"

    def save_model(self, model, scaler, metrics: Dict, best_candidates: List[Dict]):
        """Save model state, scaler, and best candidates"""
        # Save model
        with open(self.model_file, 'wb') as f:
            pickle.dump(model, f)

        # Save scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

        # Update best candidates list
        current_candidates = self.load_best_candidates()
        merged_candidates = self._merge_candidates(current_candidates, best_candidates)

        with open(self.candidates_file, 'wb') as f:
            pickle.dump(merged_candidates, f)

        # Update history
        history = self.load_history()
        history['sessions'].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'best_candidates': best_candidates,
            'total_candidates_stored': len(merged_candidates)
        })

        with open(self.history_file, 'wb') as f:
            pickle.dump(history, f)

    def load_model(self) -> tuple:
        """Load latest model and scaler if they exist"""
        model = None
        scaler = None

        if self.model_file.exists():
            with open(self.model_file, 'rb') as f:
                model = pickle.load(f)

        if self.scaler_file.exists():
            with open(self.scaler_file, 'rb') as f:
                scaler = pickle.load(f)

        return model, scaler

    def load_best_candidates(self) -> List[Dict]:
        """Load previously saved best candidates"""
        if self.candidates_file.exists():
            with open(self.candidates_file, 'rb') as f:
                return pickle.load(f)
        return []

    def _merge_candidates(self, existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge new candidates with existing ones, keeping the best unique ones"""
        # Combine lists
        all_candidates = existing + new

        # Remove duplicates and sort by X_Diff
        unique_candidates = {}
        for candidate in all_candidates:
            x_diff = float(candidate['X_Diff'])
            if x_diff not in unique_candidates or x_diff < float(unique_candidates[x_diff]['X_Diff']):
                unique_candidates[x_diff] = candidate

        # Sort by X_Diff and return top 1000 candidates
        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda x: float(x['X_Diff'])
        )
        return sorted_candidates[:1000]  # Keep top 1000 candidates

    def load_history(self) -> Dict:
        """Load training history"""
        if self.history_file.exists():
            with open(self.history_file, 'rb') as f:
                return pickle.load(f)
        return {'sessions': []}

    def get_best_candidates_ever(self) -> List[Dict]:
        """Get best candidates across all sessions"""
        return self.load_best_candidates()

    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning progress metrics over time"""
        history = self.load_history()
        metrics = {
            'mse': [],
            'r2': [],
            'min_x_diff': [],
            'candidates_count': []
        }

        for session in history['sessions']:
            for key in metrics:
                if key == 'candidates_count':
                    metrics[key].append(session.get('total_candidates_stored', 0))
                else:
                    metrics[key].append(session['metrics'].get(key, 0))

        return metrics
