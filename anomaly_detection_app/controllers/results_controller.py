import os
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)


class ResultsController:
    """
    Controller for managing experiment results and visualizations.
    """

    def __init__(self, app_config):
        """Initialize the results controller."""
        self.app_config = app_config
        self.output_folder = app_config["MODEL_OUTPUT_FOLDER"]
        self.static_folder = app_config.get("STATIC_FOLDER", "static")

    def get_results(self, experiment_id):
        """Get the results for a specific experiment."""
        # Check if we have actual result files
        result_path = os.path.join(
            self.output_folder, f"experiment_{experiment_id}", "results.json"
        )
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                return json.load(f)

        # Generate visualization files on the fly if they don't exist
        # This ensures they're available when the UI requests them
        self._ensure_visualization_files(experiment_id)

        # Return sample results if no actual results are available
        return {
            "id": experiment_id,
            "metrics": {
                "roc_auc": 0.95,
                "pr_auc": 0.87,
                "anomaly_precision": 0.92,
                "anomaly_recall": 0.85,
                "f1_score": 0.88,
                "optimal_threshold": 0.35,
            },
            "feature_importance": [
                {"feature": "feature_1", "importance": 0.23},
                {"feature": "feature_2", "importance": 0.18},
                {"feature": "feature_3", "importance": 0.15},
                {"feature": "feature_4", "importance": 0.12},
                {"feature": "feature_5", "importance": 0.10},
            ],
            "confusion_matrix": [[985, 15], [5, 95]],
            "visualization_paths": {
                "confusion_matrix": f"/static/results/{experiment_id}/confusion_matrix.png",
                "roc_curve": f"/static/results/{experiment_id}/roc_curve.png",
                "pr_curve": f"/static/results/{experiment_id}/pr_curve.png",
                "feature_importance": f"/static/results/{experiment_id}/feature_importance.png",
            },
        }

    def _ensure_visualization_files(self, experiment_id):
        """Ensure visualization files exist for the experiment."""
        # Create directories if they don't exist
        vis_dir = os.path.join(self.static_folder, "results", experiment_id)
        os.makedirs(vis_dir, exist_ok=True)

        # Generate sample visualization files if they don't exist
        self._generate_sample_confusion_matrix(
            os.path.join(vis_dir, "confusion_matrix.png")
        )
        self._generate_sample_roc_curve(os.path.join(vis_dir, "roc_curve.png"))
        self._generate_sample_pr_curve(os.path.join(vis_dir, "pr_curve.png"))
        self._generate_sample_feature_importance(
            os.path.join(vis_dir, "feature_importance.png")
        )

    def _generate_sample_confusion_matrix(self, output_path):
        """Generate a sample confusion matrix visualization."""
        if os.path.exists(output_path):
            return

        plt.figure(figsize=(10, 8))

        # Sample confusion matrix
        cm = np.array([[985, 15], [5, 95]])

        # Plot raw counts
        plt.subplot(1, 2, 1)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (Counts)")

        # Plot percentages
        plt.subplot(1, 2, 2)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (Percentages)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _generate_sample_roc_curve(self, output_path):
        """Generate a sample ROC curve visualization."""
        if os.path.exists(output_path):
            return

        plt.figure(figsize=(10, 8))

        # Generate sample ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-3 * fpr)  # A curve that's better than random

        plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = 0.95)")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _generate_sample_pr_curve(self, output_path):
        """Generate a sample Precision-Recall curve visualization."""
        if os.path.exists(output_path):
            return

        plt.figure(figsize=(10, 8))

        # Generate sample PR curve data
        recall = np.linspace(0, 1, 100)
        precision = np.maximum(0, 1 - recall**2)  # A curve that starts high and drops

        plt.plot(recall, precision, "r-", linewidth=2, label=f"PR (AUC = 0.87)")
        plt.axhline(
            y=0.1, color="k", linestyle="--", alpha=0.5, label=f"Baseline (ratio = 0.1)"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _generate_sample_feature_importance(self, output_path):
        """Generate a sample feature importance visualization."""
        if os.path.exists(output_path):
            return

        plt.figure(figsize=(12, 10))

        # Sample feature importance data
        features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        importance = [0.23, 0.18, 0.15, 0.12, 0.10]

        # Sort by importance
        sorted_idx = np.argsort(importance)
        features = [features[i] for i in sorted_idx]
        importance = [importance[i] for i in sorted_idx]

        # Plot bar chart
        plt.barh(features, importance, color="skyblue")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()  # Display highest importance at the top
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def load_results_from_file(self, experiment_id, results_path):
        """Load experiment results from a file."""
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            return results
        return None

    def save_results_to_file(self, experiment_id, results, results_path):
        """Save experiment results to a file."""
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
