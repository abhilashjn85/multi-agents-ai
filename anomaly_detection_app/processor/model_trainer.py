import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix,
                           roc_auc_score, precision_recall_curve, auc)


class ModelTrainer:
    """Class for training machine learning models"""

    def __init__(self, config):
        self.config = config

    def train_model(self, X_train, y_train, params, X_test=None, y_test=None):
        """Train XGBoost model with given parameters"""
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        eval_list = [(dtrain, 'train')]

        if X_test is not None and y_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test)
            eval_list.append((dtest, 'test'))

        # Prepare training parameters
        train_params = params.copy()
        train_params.update({
            'objective': self.config['objective'],
            'eval_metric': self.config['eval_metric']
        })

        # Train the model
        print("Training XGBoost model...")
        num_round = 500  # Maximum number of rounds
        early_stopping = 50  # Early stopping rounds

        model = xgb.train(
            train_params,
            dtrain,
            num_round,
            evals=eval_list,
            early_stopping_rounds=early_stopping,
            verbose_eval=100
        )

        print(f"Model trained for {model.best_iteration} rounds")
        print(f"Best score: {model.best_score}")

        return model

    def predict(self, model, X):
        """Make predictions with trained model"""
        dtest = xgb.DMatrix(X)
        return model.predict(dtest)

    def save_model(self, model, path):
        """Save model to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save_model(path)
        print(f"Model saved to {path}")
        return path


class ModelEvaluator:
    """Class for evaluating model performance"""

    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance with multiple metrics"""
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Calculate metrics
        report = classification_report(y_test, y_pred_binary, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        pr_auc = auc(recall, precision)

        # Print detailed evaluation
        print("\nModel Evaluation Results:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary))

        # Combine into results dictionary
        results = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'class_0': {
                'precision': report['0']['precision'],
                'recall': report['0']['recall'],
                'f1': report['0']['f1-score']
            },
            'class_1': {
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score']
            }
        }

        return results

    def interpret_results(self, results):
        """Interpret evaluation results and provide suggestions"""
        interpretation = {
            'summary': "",
            'strengths': [],
            'weaknesses': [],
            'suggestions': []
        }

        # Overall performance summary
        interpretation['summary'] = (
            f"Model achieved ROC-AUC of {results['roc_auc']:.4f} and PR-AUC of {results['pr_auc']:.4f}. "
            f"For normal cases (class 0), precision is {results['class_0']['precision']:.4f} and "
            f"recall is {results['class_0']['recall']:.4f}. "
            f"For anomaly cases (class 1), precision is {results['class_1']['precision']:.4f} and "
            f"recall is {results['class_1']['recall']:.4f}."
        )

        # Identify strengths
        if results['roc_auc'] > 0.8:
            interpretation['strengths'].append("Strong overall discriminative ability (ROC-AUC > 0.8)")

        if results['class_1']['precision'] > 0.7:
            interpretation['strengths'].append("Good precision for anomaly detection, reducing false alerts")

        if results['class_1']['recall'] > 0.7:
            interpretation['strengths'].append("Good recall for anomaly detection, catching most true anomalies")

        # Identify weaknesses
        if results['class_1']['precision'] < 0.5:
            interpretation['weaknesses'].append("Low precision for anomalies may lead to too many false alerts")

        if results['class_1']['recall'] < 0.5:
            interpretation['weaknesses'].append("Low recall for anomalies means many anomalies may be missed")

        if abs(results['class_1']['precision'] - results['class_1']['recall']) > 0.3:
            interpretation['weaknesses'].append("Large imbalance between precision and recall for anomalies")

        # Make suggestions
        if results['class_1']['precision'] < 0.5:
            interpretation['suggestions'].append(
                "Consider adjusting the classification threshold to improve precision"
            )

        if results['class_1']['recall'] < 0.5:
            interpretation['suggestions'].append(
                "Try increasing the anomaly ratio in training data to improve recall"
            )

        if results['roc_auc'] < 0.7:
            interpretation['suggestions'].append(
                "Model may benefit from additional feature engineering or different algorithm"
            )

        # Print interpretation
        print("\nModel Interpretation:")
        print(interpretation['summary'])

        print("\nStrengths:")
        for strength in interpretation['strengths']:
            print(f"• {strength}")

        print("\nWeaknesses:")
        for weakness in interpretation['weaknesses']:
            print(f"• {weakness}")

        print("\nSuggestions:")
        for suggestion in interpretation['suggestions']:
            print(f"• {suggestion}")

        return interpretation

    def find_optimal_threshold(self, model, X_test, y_test):
        """Find optimal classification threshold for imbalanced data"""
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)

        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []

        print("\nFinding optimal classification threshold:")
        print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'F2':<10}")
        print("-" * 50)

        for threshold in thresholds:
            y_pred_binary = (y_pred >= threshold).astype(int)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)

            # F2 score gives more weight to recall
            beta = 2
            f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0

            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'f2': f2
            })

            print(f"{threshold:<10.2f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {f2:<10.4f}")

        # Find optimal threshold based on F1 score
        f1_optimal = max(results, key=lambda x: x['f1'])
        # Find optimal threshold based on F2 score
        f2_optimal = max(results, key=lambda x: x['f2'])

        print("\nOptimal threshold based on F1 score:")
        print(f"Threshold: {f1_optimal['threshold']:.2f}, Precision: {f1_optimal['precision']:.4f}, " +
              f"Recall: {f1_optimal['recall']:.4f}, F1: {f1_optimal['f1']:.4f}")

        print("\nOptimal threshold based on F2 score (prioritizes recall):")
        print(f"Threshold: {f2_optimal['threshold']:.2f}, Precision: {f2_optimal['precision']:.4f}, " +
             f"Recall: {f2_optimal['recall']:.4f}, F2: {f2_optimal['f2']:.4f}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        return {
            'f1_optimal': f1_optimal['threshold'],
            'f2_optimal': f2_optimal['threshold'],
            'results': results_df
        }


def save_model_artifacts(model, vectorizer, mlb, config, label_encoders, metrics, model_dir):
    """Save model and related artifacts"""
    import os
    import joblib

    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, 'xgboost_model.json')
    model.save_model(model_path)

    # Save vectorizer and mlb
    if vectorizer is not None:
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        joblib.dump(vectorizer, vectorizer_path)

    if mlb is not None:
        mlb_path = os.path.join(model_dir, 'mlb.joblib')
        joblib.dump(mlb, mlb_path)

    # Save label encoders
    if label_encoders:
        le_path = os.path.join(model_dir, 'label_encoders.joblib')
        joblib.dump(label_encoders, le_path)

    # Save config
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        if isinstance(config, dict):
            json.dump(config, f, indent=4)
        else:
            # Handle Config object by converting to dict
            json.dump(config.to_dict() if hasattr(config, 'to_dict') else config.__dict__, f, indent=4)

    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        metrics_serializable = {}
        for k, v in metrics.items():
            if k not in ['confusion_matrix', 'classification_report']:
                metrics_serializable[k] = v
        # Add serialized confusion matrix and classification report
        if 'confusion_matrix' in metrics:
            if isinstance(metrics['confusion_matrix'], np.ndarray):
                metrics_serializable['confusion_matrix'] = metrics['confusion_matrix'].tolist()
            else:
                metrics_serializable['confusion_matrix'] = metrics['confusion_matrix']
        if 'classification_report' in metrics:
            metrics_serializable['classification_report'] = metrics['classification_report']

        # Ensure JSON serializable
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        metrics_serializable = convert_to_serializable(metrics_serializable)
        json.dump(metrics_serializable, f, indent=4)

    print(f"Model and artifacts saved to {model_dir}")
    return {
        'model_path': model_path,
        'vectorizer_path': os.path.join(model_dir, 'tfidf_vectorizer.joblib') if vectorizer is not None else None,
        'mlb_path': os.path.join(model_dir, 'mlb.joblib') if mlb is not None else None,
        'le_path': os.path.join(model_dir, 'label_encoders.joblib') if label_encoders else None,
        'config_path': config_path,
        'metrics_path': metrics_path
    }