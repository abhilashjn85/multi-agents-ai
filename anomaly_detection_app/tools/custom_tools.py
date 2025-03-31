# Add these at the top of your file or in a separate tools.py module
from crewai.tools import BaseTool, tool
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import os

from pydantic import Field, ConfigDict
from typing import Any, Dict


class DataLoaderTool(BaseTool):
    """Tool for loading data from file"""
    name: str = "data_loader"
    description: str = "Load data from a file and provide basic statistics"
    # Define model_config to ignore methods decorated with
    experiment: Any = None  # Add this field explicitly
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None):
        super().__init__()
        self.experiment = experiment  # Store experiment here

    def _run(self, data_path: str) -> dict:
        """
        Load data from the specified file path and return statistics.

        Args:
            data_path: Path to the data file (CSV or Excel)

        Returns:
            Dictionary with data statistics
        """
        try:
            # Load data
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                data = pd.read_excel(data_path)
            else:
                return {"error": f"Unsupported file format: {data_path}"}

            # Calculate basic statistics
            stats = {
                "shape": str(data.shape),
                "columns": data.columns.tolist(),
                "missing_values": data.isnull().sum().to_dict(),
                "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()}
            }

            if 'IS_ANOMALY' in data.columns:
                anomaly_count = data['IS_ANOMALY'].sum()
                anomaly_ratio = anomaly_count / len(data)
                stats["anomaly_count"] = int(anomaly_count)
                stats["anomaly_ratio"] = float(anomaly_ratio)

            # Store the raw data in experiment if available
            if self.experiment:
                self.experiment.raw_data = data

            return stats
        except Exception as e:
            return {"error": str(e)}


class DataProcessorTool(BaseTool):
    """Tool for processing data"""
    name: str = "data_processor"
    description: str = "Process raw data according to configuration rules"
    experiment: Any = None
    config: Any = None
    processor_class: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None, config=None, processor_class=None):
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.processor_class = processor_class

    def _run(self) -> dict:
        """
        Process raw data according to configuration.

        Returns:
            Dictionary with processing results
        """
        try:
            # Get data from experiment storage
            if not self.experiment or not hasattr(self.experiment, 'raw_data') or self.experiment.raw_data is None:
                return {"error": "No data loaded. Use data_loader first."}

            raw_data = self.experiment.raw_data

            # Create processor
            processor = self.processor_class(self.config)

            # Process data
            processed_data, label_encoders = processor.process_data(raw_data)

            # Store in experiment storage
            if self.experiment:
                self.experiment.processed_data = processed_data
                self.experiment.label_encoders = label_encoders

            # Create statistics for reporting
            stats = {
                "processed_shape": str(processed_data.shape),
                "new_columns": [col for col in processed_data.columns if col not in raw_data.columns],
                "encoded_columns": [col for col in processed_data.columns if col.endswith('_encoded')],
                "anomaly_columns": [col for col in processed_data.columns if '_anomaly_' in col]
            }

            return stats
        except Exception as e:
            return {"error": str(e)}


class FeatureEngineeringTool(BaseTool):
    """Tool for feature engineering"""
    name: str = "feature_engineer"
    description: str = "Create features from processed data"
    experiment: Any = None
    config: Any = None
    engineer_class: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None, config=None, engineer_class=None):
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.engineer_class = engineer_class

    def _run(self) -> dict:
        """
        Create features from processed data.

        Returns:
            Dictionary with feature engineering results
        """
        try:
            # Get processed data from experiment storage
            if not self.experiment or not hasattr(self.experiment,
                                                  'processed_data') or self.experiment.processed_data is None:
                return {"error": "No processed data. Use data_processor first."}

            processed_data = self.experiment.processed_data

            # Create engineer
            engineer = self.engineer_class(self.config)

            # Create features
            vectorizer, mlb, features, feature_stats = engineer.create_features(processed_data)

            # Create feature names
            feature_names = []
            if vectorizer:
                vocab = vectorizer.get_feature_names_out()
                feature_names.extend([f"tfidf_{word}" for word in vocab])

            for col in self.config["categorical_columns"]:
                if f"{col}_encoded" in processed_data.columns:
                    feature_names.append(f"cat_{col}")

            if mlb and hasattr(mlb, 'classes_'):
                for cls in mlb.classes_:
                    feature_names.append(f"multi_{cls}")

            for col in processed_data.columns:
                if "_anomaly_" in col:
                    feature_names.append(col)

            # Make sure we have enough feature names
            while len(feature_names) < features.shape[1]:
                feature_names.append(f"feature_{len(feature_names)}")

            # Extract labels if available
            labels = None
            if 'IS_ANOMALY' in processed_data.columns:
                labels = processed_data['IS_ANOMALY'].values

            # Store in experiment storage
            if self.experiment:
                self.experiment.features = features
                self.experiment.feature_names = feature_names
                self.experiment.labels = labels
                self.experiment.vectorizer = vectorizer
                self.experiment.mlb = mlb

            # Create statistics for reporting
            stats = {
                "feature_shape": str(features.shape),
                "feature_density": float(np.mean(features != 0)),
                "feature_sparsity": float(np.mean(features == 0)),
                "num_features": features.shape[1],
                "has_labels": labels is not None
            }

            return stats
        except Exception as e:
            return {"error": str(e)}


class DataSplitterTool(BaseTool):
    """Tool for data splitting"""
    name: str = "data_splitter"
    experiment: Any = None  # Add this field explicitly
    description: str = "Split data for training and testing"
    config: Any = None
    splitter_class: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None, config=None, splitter_class=None):
        super().__init__()
        self.experiment = experiment  # Store experiment here
        self.config = config
        self.splitter_class = splitter_class

    def _run(self) -> dict:
        """
        Split data into training and testing sets.

        Returns:
            Dictionary with split results
        """
        try:
            # Get features and labels from experiment storage
            if not self.experiment or not hasattr(self.experiment, 'features') or self.experiment.features is None:
                return {"error": "No features or labels. Use feature_engineer first."}

            features = self.experiment.features
            labels = self.experiment.labels

            # Create splitter
            splitter = self.splitter_class(self.config)

            # Find optimal anomaly ratio
            best_ratio, ratio_results = splitter.find_optimal_anomaly_ratio(
                features, labels, test_size=0.2, ratios=[0.01, 0.05, 0.1, 0.15, 0.2]
            )

            # Split the data
            X_train, X_test, y_train, y_test = splitter.custom_train_test_split(
                features, labels, test_size=0.2, anomaly_ratio=best_ratio
            )

            # Store in experiment storage
            if self.experiment:
                self.experiment.X_train = X_train
                self.experiment.X_test = X_test
                self.experiment.y_train = y_train
                self.experiment.y_test = y_test

            # Create statistics for reporting
            stats = {
                "best_anomaly_ratio": best_ratio,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_anomaly_ratio": float(np.mean(y_train)),
                "test_anomaly_ratio": float(np.mean(y_test)),
                "ratios_tested": [0.01, 0.05, 0.1, 0.15, 0.2]
            }

            return stats
        except Exception as e:
            return {"error": str(e)}


class ModelOptimizerTool(BaseTool):
    """Tool for model optimization"""
    name: str = "model_optimizer"
    experiment: Any = None  # Add this field explicitly
    description: str = "Find optimal hyperparameters using genetic algorithm"
    config: Any = None
    optimizer_class: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None, config=None, optimizer_class=None):
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.optimizer_class = optimizer_class

    def _run(self) -> dict:
        """
        Find optimal hyperparameters for the model using genetic algorithm.

        Returns:
            Dictionary with optimization results
        """
        try:
            # Get training data from experiment storage
            if not self.experiment or not hasattr(self.experiment, 'X_train') or self.experiment.X_train is None:
                return {"error": "No training data. Use data_splitter first."}

            X_train = self.experiment.X_train
            y_train = self.experiment.y_train

            # Create optimizer
            optimizer = self.optimizer_class(self.config, X_train, y_train)

            # Run optimization
            best_params, best_fitness, fitness_history = optimizer.optimize()

            # Store in experiment storage
            if self.experiment:
                self.experiment.best_params = best_params

            # Convert to simple types for reporting
            best_params_simple = {k: float(v) if isinstance(v, (np.number, float)) else v
                                  for k, v in best_params.items()}

            return {
                "best_params": best_params_simple,
                "best_fitness": float(best_fitness),
                "generations": len(fitness_history) if fitness_history else 0
            }
        except Exception as e:
            return {"error": str(e)}


class ModelTrainerTool(BaseTool):
    """Tool for model training"""
    name: str = "model_trainer"
    experiment: Any = None  # Add this field explicitly
    description: str = "Train XGBoost model with optimal parameters"
    config: Any = None
    trainer_class: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None, config=None, trainer_class=None):
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.trainer_class = trainer_class
    
    def _run(self) -> dict:
        """
        Train an XGBoost model with the provided parameters.

        Returns:
            Dictionary with training results
        """
        try:
            # Get data from experiment storage
            if not self.experiment:
                return {"error": "No experiment provided"}

            X_train = self.experiment.X_train
            y_train = self.experiment.y_train
            X_test = self.experiment.X_test
            y_test = self.experiment.y_test
            best_params = self.experiment.best_params

            if X_train is None or y_train is None:
                return {"error": "No training data. Use data_splitter first."}

            if best_params is None:
                return {"error": "No model parameters. Use model_optimizer first."}

            # Create trainer
            trainer = self.trainer_class(self.config)

            # Train model
            model = trainer.train_model(X_train, y_train, best_params, X_test, y_test)

            # Store in experiment storage
            if self.experiment:
                self.experiment.model = model

            # Return results
            return {
                "best_iteration": int(model.best_iteration),
                "best_score": float(model.best_score),
                "num_features": X_train.shape[1],
                "model_type": "XGBoost"
            }
        except Exception as e:
            return {"error": str(e)}


class ModelEvaluatorTool(BaseTool):
    """Tool for model evaluation"""
    name: str = "model_evaluator"
    experiment: Any = None  # Add this field explicitly
    description: str = "Evaluate model performance with appropriate metrics"
    config: Any = None
    evaluator_class: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None, config=None, evaluator_class=None):
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.evaluator_class = evaluator_class

    def _run(self) -> dict:
        """
        Evaluate model performance on test data.

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get data from experiment storage
            if not self.experiment:
                return {"error": "No experiment provided"}

            model = self.experiment.model
            X_test = self.experiment.X_test
            y_test = self.experiment.y_test

            if model is None or X_test is None or y_test is None:
                return {"error": "Model or test data not available. Use train_model first."}

            # Create evaluator
            evaluator = self.evaluator_class(self.config)

            # Evaluate model
            evaluation_results = evaluator.evaluate_model(model, X_test, y_test)

            # Find optimal threshold
            threshold_results = evaluator.find_optimal_threshold(model, X_test, y_test)

            # Add threshold results to evaluation results
            evaluation_results['optimal_thresholds'] = threshold_results

            # Store in experiment storage
            if self.experiment:
                self.experiment.evaluation_results = evaluation_results

            # Convert complex types to simple types for reporting
            simple_results = {
                "roc_auc": float(evaluation_results['roc_auc']),
                "pr_auc": float(evaluation_results['pr_auc']),
                "anomaly_precision": float(evaluation_results['class_1']['precision']),
                "anomaly_recall": float(evaluation_results['class_1']['recall']),
                "anomaly_f1": float(evaluation_results['class_1']['f1']),
                "optimal_threshold_f1": float(threshold_results['f1_optimal']),
                "optimal_threshold_f2": float(threshold_results['f2_optimal']),
                "confusion_matrix": str(evaluation_results['confusion_matrix'])
            }

            return simple_results
        except Exception as e:
            return {"error": str(e)}


class FeatureAnalyzerTool(BaseTool):
    """Tool for feature analysis"""
    name: str = "feature_analyzer"
    description: str = "Analyze feature importance and suggest improvements"
    model_config = ConfigDict(arbitrary_types_allowed=True)
    experiment: Any = Field(default=None)
    config: Dict = Field(default_factory=dict)
    engineer_class: Any = Field(default=None)

    def __init__(self, experiment, config, engineer_class):
        super().__init__()
        self.experiment = experiment  # Store experiment here
        self.config = config
        self.engineer_class = engineer_class

    def _run(self) -> dict:
        """
        Analyze feature importance and suggest improvements.

        Returns:
            Dictionary with feature analysis
        """
        try:
            # Get data from experiment storage
            if not self.experiment:
                return {"error": "No experiment provided"}

            model = self.experiment.model
            feature_names = self.experiment.feature_names
            processed_data = self.experiment.processed_data

            if model is None or feature_names is None or processed_data is None:
                return {"error": "Model, feature names, or processed data not available."}

            # Create engineer
            engineer = self.engineer_class(self.config)

            # Analyze feature importance
            top_n = 20
            feature_importance = engineer.analyze_feature_importance(model, feature_names, top_n)

            # Get feature suggestions
            feature_suggestions = engineer.suggest_features(processed_data, feature_importance)

            # Store in experiment storage
            if self.experiment:
                self.experiment.feature_importance = feature_importance

            # Convert to simple format for reporting
            top_features = []
            if hasattr(feature_importance, 'to_dict'):
                for i, row in feature_importance.iterrows():
                    top_features.append({
                        "feature": row['Feature'],
                        "importance": float(row['Importance'])
                    })

            return {
                "top_features": top_features,
                "suggestions": feature_suggestions
            }
        except Exception as e:
            return {"error": str(e)}


class QualityAssessmentTool(BaseTool):
    """Tool for quality assessment"""
    name: str = "quality_assessment"
    description: str = "Assess model quality and provide recommendations"
    config: Any = None
    experiment: Any = None
    evaluator_class: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, experiment=None, config=None, evaluator_class=None):
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.evaluator_class = evaluator_class
    
    def _run(self) -> dict:
        """
        Assess model quality and provide recommendations.

        Returns:
            Dictionary with quality assessment
        """
        try:
            # Get evaluation results from experiment storage
            if not self.experiment or not hasattr(self.experiment, 'evaluation_results'):
                return {"error": "No evaluation results available."}

            evaluation_results = self.experiment.evaluation_results

            if evaluation_results is None:
                return {"error": "Evaluation results not available."}

            # Create evaluator
            evaluator = self.evaluator_class(self.config)

            # Interpret results
            interpretation = evaluator.interpret_results(evaluation_results)

            # Store in experiment storage
            if self.experiment:
                self.experiment.quality_assessment = interpretation

            return {
                "summary": interpretation['summary'],
                "strengths": interpretation['strengths'],
                "weaknesses": interpretation['weaknesses'],
                "suggestions": interpretation['suggestions'],
                "recommendation": "go" if len(interpretation['strengths']) > len(
                    interpretation['weaknesses']) else "no-go"
            }
        except Exception as e:
            return {"error": str(e)}


class ModelSaverTool(BaseTool):
    """Tool for saving model artifacts"""
    name: str = "model_saver"
    description: str = "Save model artifacts and results"
    model_config = ConfigDict(arbitrary_types_allowed=True)
    save_function: Any = None
    experiment: Any = None

    def __init__(self, experiment=None, save_function=None):
        super().__init__()
        self.experiment = experiment
        self.save_function = save_function
    
    def _run(self) -> dict:
        """
        Save model artifacts and results.

        Returns:
            Dictionary with paths to saved artifacts
        """
        try:
            # Get data from experiment storage
            if not self.experiment:
                return {"error": "No experiment provided"}

            model = self.experiment.model
            vectorizer = self.experiment.vectorizer
            mlb = self.experiment.mlb
            label_encoders = self.experiment.label_encoders
            evaluation_results = self.experiment.evaluation_results
            quality_assessment = self.experiment.quality_assessment
            config = self.experiment.config_dict if hasattr(self.experiment, 'config_dict') else None

            if model is None:
                return {"error": "Model not available."}

            # Create metrics dictionary
            metrics = {
                'roc_auc': evaluation_results['roc_auc'] if evaluation_results else None,
                'pr_auc': evaluation_results['pr_auc'] if evaluation_results else None,
                'optimal_threshold_f1': evaluation_results['optimal_thresholds'][
                    'f1_optimal'] if evaluation_results else None,
                'optimal_threshold_f2': evaluation_results['optimal_thresholds'][
                    'f2_optimal'] if evaluation_results else None,
                'anomaly_precision': evaluation_results['class_1']['precision'] if evaluation_results else None,
                'anomaly_recall': evaluation_results['class_1']['recall'] if evaluation_results else None,
                'anomaly_f1': evaluation_results['class_1']['f1'] if evaluation_results else None,
                'classification_report': evaluation_results['classification_report'] if evaluation_results else None,
                'confusion_matrix': evaluation_results['confusion_matrix'] if evaluation_results else None,
                'interpretation': quality_assessment if quality_assessment else None
            }

            # Save artifacts
            artifact_paths = self.save_function(
                model, vectorizer, mlb, config,
                label_encoders, metrics, self.experiment.output_path
            )

            return {
                "model_path": artifact_paths.get('model_path', ''),
                "vectorizer_path": artifact_paths.get('vectorizer_path', ''),
                "config_path": artifact_paths.get('config_path', ''),
                "metrics_path": artifact_paths.get('metrics_path', '')
            }
        except Exception as e:
            return {"error": str(e)}
