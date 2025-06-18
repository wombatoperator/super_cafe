"""
Manifest for tracking feature generation results and persistence.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class Manifest:
    """
    Tracks and persists feature generation results.
    
    Maintains a record of all generated features, their performance,
    and metadata for reproducibility and analysis.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize manifest.
        
        Args:
            session_id: Optional session identifier. If None, generates a new UUID.
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.records: List[Dict[str, Any]] = []
        self.metadata = {
            "session_id": self.session_id,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0",
            "total_iterations": 0,
            "successful_features": 0,
            "failed_attempts": 0,
        }
    
    def add_iteration(
        self,
        iteration: int,
        code: str,
        explanation: str,
        delta_auc: float,
        fold_scores: List[float],
        feature_name: str,
        execution_time: float,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> None:
        """
        Add a feature generation iteration to the manifest.
        
        Args:
            iteration: Iteration number
            code: Generated code
            explanation: LLM explanation of the feature
            delta_auc: AUC improvement (can be negative)
            fold_scores: Individual fold scores
            feature_name: Name of the generated feature
            execution_time: Time taken for evaluation
            status: "success", "failed", or "rejected"
            error_message: Error message if failed
        """
        record = {
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "code": code,
            "explanation": explanation,
            "feature_name": feature_name,
            "delta_auc": delta_auc,
            "fold_scores": fold_scores,
            "mean_fold_score": float(pd.Series(fold_scores).mean()) if fold_scores else None,
            "std_fold_score": float(pd.Series(fold_scores).std()) if fold_scores else None,
            "execution_time": execution_time,
            "status": status,
            "error_message": error_message,
            "accepted": delta_auc > 0 if status == "success" else False,
        }
        
        self.records.append(record)
        
        # Update metadata
        self.metadata["total_iterations"] = len(self.records)
        if status == "success" and delta_auc > 0:
            self.metadata["successful_features"] += 1
        elif status == "failed":
            self.metadata["failed_attempts"] += 1
        
        self.metadata["last_updated"] = datetime.utcnow().isoformat()
    
    def get_accepted_features(self) -> List[Dict[str, Any]]:
        """Get all accepted features (positive delta_auc)."""
        return [r for r in self.records if r["accepted"]]
    
    def get_best_feature(self) -> Optional[Dict[str, Any]]:
        """Get the feature with the highest AUC improvement."""
        accepted = self.get_accepted_features()
        if not accepted:
            return None
        return max(accepted, key=lambda x: x["delta_auc"])
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        if not self.records:
            return {"total_iterations": 0, "success_rate": 0.0}
        
        accepted = self.get_accepted_features()
        total_improvement = sum(r["delta_auc"] for r in accepted)
        
        return {
            "total_iterations": len(self.records),
            "successful_features": len(accepted),
            "success_rate": len(accepted) / len(self.records),
            "total_improvement": total_improvement,
            "avg_improvement": total_improvement / len(accepted) if accepted else 0.0,
            "best_improvement": max(r["delta_auc"] for r in accepted) if accepted else 0.0,
            "execution_time_total": sum(r["execution_time"] for r in self.records),
            "avg_execution_time": sum(r["execution_time"] for r in self.records) / len(self.records),
        }
    
    def get_feature_code(self, feature_name: str) -> Optional[str]:
        """Get the code for a specific feature."""
        for record in self.records:
            if record["feature_name"] == feature_name and record["accepted"]:
                return record["code"]
        return None
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save manifest to JSON file.
        
        Args:
            path: Path to save the manifest
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest_data = {
            "metadata": self.metadata,
            "summary": self.get_summary_stats(),
            "records": self.records,
        }
        
        with open(path, "w") as f:
            json.dump(manifest_data, f, indent=2, default=str)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load manifest from JSON file.
        
        Args:
            path: Path to load the manifest from
        """
        with open(path, "r") as f:
            data = json.load(f)
        
        self.metadata = data["metadata"]
        self.records = data["records"]
        self.session_id = self.metadata["session_id"]
    
    def export_features_dataframe(self) -> pd.DataFrame:
        """Export accepted features as a DataFrame for analysis."""
        accepted = self.get_accepted_features()
        if not accepted:
            return pd.DataFrame()
        
        df_data = []
        for record in accepted:
            df_data.append({
                "iteration": record["iteration"],
                "feature_name": record["feature_name"],
                "delta_auc": record["delta_auc"],
                "mean_fold_score": record["mean_fold_score"],
                "std_fold_score": record["std_fold_score"],
                "execution_time": record["execution_time"],
                "timestamp": record["timestamp"],
                "explanation": record["explanation"][:100] + "..." if len(record["explanation"]) > 100 else record["explanation"],
            })
        
        return pd.DataFrame(df_data)
    
    def export_code_file(self, path: Union[str, Path]) -> None:
        """
        Export all accepted feature code to a Python file.
        
        Args:
            path: Path to save the code file
        """
        accepted = self.get_accepted_features()
        if not accepted:
            return
        
        with open(path, "w") as f:
            f.write("# Generated features from CAAFE\n")
            f.write(f"# Session: {self.session_id}\n")
            f.write(f"# Generated: {datetime.utcnow().isoformat()}\n\n")
            f.write("import pandas as pd\nimport numpy as np\n\n")
            
            for i, record in enumerate(accepted, 1):
                f.write(f"# Feature {i}: {record['feature_name']}\n")
                f.write(f"# AUC Improvement: +{record['delta_auc']:.4f}\n")
                f.write(f"# Explanation: {record['explanation']}\n")
                f.write(f"{record['code']}\n\n")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history for LLM context."""
        history = []
        for record in self.records:
            if record["status"] == "success":
                if record["accepted"]:
                    feedback = f"Great! ROC_AUC +{record['delta_auc']:.4f}. Next idea?"
                else:
                    feedback = f"Δ={record['delta_auc']:.4f} ≤0. Try alt logic."
            else:
                feedback = f"Code rejected: {record['error_message']}"
            
            history.append({
                "iteration": record["iteration"],
                "explanation": record["explanation"],
                "feedback": feedback
            })
        
        return history
    
    def __str__(self) -> str:
        """String representation of the manifest."""
        stats = self.get_summary_stats()
        return (
            f"Manifest {self.session_id}: "
            f"{stats['total_iterations']} iterations, "
            f"{stats['successful_features']} successful features, "
            f"{stats['success_rate']:.1%} success rate"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of the manifest."""
        return f"Manifest(session_id='{self.session_id}', records={len(self.records)})"


class ManifestManager:
    """
    Manages multiple manifests and provides utilities for analysis.
    """
    
    def __init__(self, base_dir: Union[str, Path] = "manifests"):
        """
        Initialize manifest manager.
        
        Args:
            base_dir: Base directory for storing manifests
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_manifest(self, session_id: Optional[str] = None) -> Manifest:
        """Create a new manifest."""
        return Manifest(session_id=session_id)
    
    def save_manifest(self, manifest: Manifest, filename: Optional[str] = None) -> Path:
        """
        Save a manifest to the managed directory.
        
        Args:
            manifest: Manifest to save
            filename: Optional filename (defaults to session_id.json)
            
        Returns:
            Path where the manifest was saved
        """
        if filename is None:
            filename = f"{manifest.session_id}.json"
        
        path = self.base_dir / filename
        manifest.save(path)
        return path
    
    def load_manifest(self, filename: str) -> Manifest:
        """Load a manifest from the managed directory."""
        path = self.base_dir / filename
        manifest = Manifest()
        manifest.load(path)
        return manifest
    
    def list_manifests(self) -> List[str]:
        """List all manifest files in the managed directory."""
        return [f.name for f in self.base_dir.glob("*.json")]
    
    def get_best_features_across_sessions(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get the best features across all saved manifests.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            DataFrame with best features
        """
        all_features = []
        
        for manifest_file in self.list_manifests():
            try:
                manifest = self.load_manifest(manifest_file)
                features_df = manifest.export_features_dataframe()
                if not features_df.empty:
                    features_df["session_id"] = manifest.session_id
                    all_features.append(features_df)
            except Exception as e:
                print(f"Error loading manifest {manifest_file}: {e}")
                continue
        
        if not all_features:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_features, ignore_index=True)
        return combined_df.nlargest(top_k, "delta_auc")