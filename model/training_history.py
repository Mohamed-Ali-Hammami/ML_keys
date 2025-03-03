import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class TrainingHistory:
    def __init__(self, history_file: str = "logs/training_history.json"):
        self.history_file = history_file
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ensure_history_file()
        
    def ensure_history_file(self):
        """Create history file and directory if they don't exist"""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump({"sessions": []}, f)

    def save_training_metrics(self, metrics: Dict, params: Dict, best_candidates: List[Dict]):
        """Save training metrics and best candidates for this session"""
        history = self.load_history()
        
        # Convert numpy types to Python native types
        metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                  for k, v in metrics.items()}
        
        session_data = {
            "session_id": self.current_session,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "parameters": params,
            "best_candidates": best_candidates
        }
        
        history["sessions"].append(session_data)
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def load_history(self) -> Dict:
        """Load training history"""
        with open(self.history_file, 'r') as f:
            return json.load(f)

    def get_progress_metrics(self) -> pd.DataFrame:
        """Get progress metrics across all sessions"""
        history = self.load_history()
        metrics_list = []
        
        for session in history["sessions"]:
            metrics = session["metrics"]
            metrics["session_id"] = session["session_id"]
            metrics["timestamp"] = session["timestamp"]
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list)

    def get_best_candidates_ever(self, n: int = 10) -> List[Dict]:
        """Get the best n candidates across all sessions"""
        history = self.load_history()
        all_candidates = []
        
        for session in history["sessions"]:
            all_candidates.extend(session["best_candidates"])
        
        return sorted(all_candidates, key=lambda x: float(x["X_Diff"]))[:n]

    def get_parameter_evolution(self) -> pd.DataFrame:
        """Track how parameters changed across sessions"""
        history = self.load_history()
        params_list = []
        
        for session in history["sessions"]:
            params = session["parameters"]
            params["session_id"] = session["session_id"]
            params["timestamp"] = session["timestamp"]
            params_list.append(params)
        
        return pd.DataFrame(params_list)
