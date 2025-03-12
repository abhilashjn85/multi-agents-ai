import os
import json
import uuid
import datetime
import threading
import time
import pandas as pd
import numpy as np
from anomaly_detection_app.models.experiment import Experiment, ExperimentConfig
from anomaly_detection_app.models.workflow import Workflow

# Import ML components for actual logic execution
from anomaly_detection_app.processor.data_processor import DataProcessor, FeatureEngineer
from anomaly_detection_app.processor.data_splitter import DataSplitter, GAOptimizer
from anomaly_detection_app.processor.model_trainer import ModelTrainer, ModelEvaluator, save_model_artifacts

# Import our direct Mistral client
from anomaly_detection_app.models.mistral_client import MistralClient, AgentRunner


class ExperimentController:
    """
    Controller for managing experiments using direct Mistral API calls
    and executing the actual ML pipeline.
    """

    def __init__(self, app_config):
        """Initialize the experiment controller."""
        self.app_config = app_config
        self.experiments = {}
        self.configs = {}
        self.active_runs = {}
        self.data_path = ""
        self.output_path = app_config['MODEL_OUTPUT_FOLDER']
        self.agent_controller = None
        self.workflow_controller = None

        # Get the LLM API URL from configuration
        self.llm_api_url = app_config.get('LLM_API_URL',
                                          'https://aiplatform.dev51.cbf.dev.paypalinc.com/seldon/seldon/mistral-7b-inst-2252b/v2/models/mistral-7b-inst-2252b/infer')

        # Load default configuration
        self._load_default_config()

    def register_controllers(self, agent_controller, workflow_controller):
        """Register dependencies on other controllers"""
        self.agent_controller = agent_controller
        self.workflow_controller = workflow_controller

    def _load_default_config(self):
        """Load the default experiment configuration."""
        default_config = ExperimentConfig.default_config()
        self.configs[default_config.id] = default_config

    def get_config(self, config_id=None):
        """Get a specific configuration or the default if no ID is provided."""
        if config_id and config_id in self.configs:
            return self.configs[config_id].to_dict()
        elif self.configs:
            # Return the first config (default)
            return next(iter(self.configs.values())).to_dict()
        return None

    def get_configs(self):
        """Get all available configurations."""
        return [config.to_dict() for config in self.configs.values()]

    def update_config(self, data):
        """Update a configuration."""
        config_id = data.get('id')
        if config_id and config_id in self.configs:
            # Update the existing config
            config = self.configs[config_id]
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config.to_dict()
        else:
            # Create a new config
            config = ExperimentConfig.from_dict(data)
            self.configs[config.id] = config
            return config.to_dict()

    def load_config(self, file_path):
        """Load a configuration from a JSON file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config_data = json.load(f)

            # Create a new config from the file
            config = ExperimentConfig.from_dict(config_data)
            self.configs[config.id] = config
            return config.to_dict()
        return None

    def set_data_path(self, file_path):
        """Set the data path for experiments."""
        self.data_path = file_path

    def get_experiments(self):
        """Get all experiments."""
        return [experiment.to_dict() for experiment in self.experiments.values()]

    def get_recent_experiments(self, limit=5):
        """Get the most recent experiments."""
        experiments = sorted(
            self.experiments.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return [exp.to_dict() for exp in experiments[:limit]]

    def get_experiment(self, experiment_id):
        """Get a specific experiment by ID."""
        if experiment_id in self.experiments:
            return self.experiments[experiment_id].to_dict()
        return None

    def get_experiment_status(self, experiment_id):
        """Get the status of an experiment."""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            return {
                'id': experiment.id,
                'status': experiment.status,
                'progress': experiment.progress,
                'current_phase': experiment.current_phase,
                'current_agent': experiment.current_agent,
                'started_at': experiment.started_at,
                'completed_at': experiment.completed_at,
                'recent_logs': experiment.log_entries[-10:] if experiment.log_entries else []
            }
        return None

    def create_experiment(self, workflow_id, config_id):
        """Create a new experiment."""
        # Generate a unique ID for the experiment
        experiment_id = str(uuid.uuid4())

        # Generate a unique output path for this experiment
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_path, f"experiment_{timestamp}")
        os.makedirs(output_path, exist_ok=True)

        # Create the experiment object
        experiment = Experiment(
            id=experiment_id,
            name=f"Experiment {timestamp}",
            description="Anomaly detection experiment",
            workflow_id=workflow_id,
            config_id=config_id,
            data_path=self.data_path,
            output_path=output_path,
            created_at=datetime.datetime.now().isoformat(),
            status="created"
        )

        self.experiments[experiment_id] = experiment
        return experiment.to_dict()

    # [standard methods for config and experiment management]
    # ... all the standard methods remain unchanged ...

    def run_experiment(self, experiment_id):
        """Run an experiment in a background thread."""
        if experiment_id not in self.experiments:
            return None

        experiment = self.experiments[experiment_id]

        # Create a thread to run the experiment
        thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(experiment,)
        )
        thread.daemon = True
        thread.start()

        # Add to active runs
        self.active_runs[experiment_id] = thread

        return experiment.to_dict()

    def _run_experiment_thread(self, experiment):
        """Background thread that runs the experiment."""
        try:
            # Update experiment status
            experiment.update_status("running", 0.0, "initialization", "")
            experiment.add_log_entry("Starting experiment run", level="INFO")

            # Get workflow and config
            workflow_id = experiment.workflow_id
            config_id = experiment.config_id

            # Check if workflow and config exist in their respective controllers
            # For a real implementation, this would access the controllers directly
            # For now, we'll simulate the workflow execution

            # Load the original code components dynamically
            self.run_actual_experiment(experiment, self.agent_controller, self.workflow_controller)
            # self._simulate_experiment_run(experiment)
        except Exception as e:
            experiment.update_status("failed")
            experiment.add_log_entry(
                f"Error running experiment: {str(e)}", level="ERROR"
            )

    def run_actual_experiment(self, experiment, agent_controller, workflow_controller):
        """
        Run the actual experiment using direct Mistral API calls AND the ML pipeline.
        """
        try:
            # Update experiment status
            experiment.update_status("running", 0.1, "initialization", "")
            experiment.add_log_entry("Starting experiment using direct Mistral API", level="INFO")

            # Get the workflow and config
            workflow_dict = workflow_controller.get_workflow(experiment.workflow_id)
            config_dict = self.get_config(experiment.config_id)

            if not workflow_dict or not config_dict:
                experiment.update_status("failed")
                experiment.add_log_entry("Workflow or config not found", level="ERROR")
                return

            # Save config to a temporary file
            config_path = os.path.join(experiment.output_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            experiment.add_log_entry(f"Using data path: {experiment.data_path}", level="INFO")
            experiment.add_log_entry(f"Config saved to: {config_path}", level="INFO")

            # PHASE 1: Run agent dialogue using direct Mistral API
            agent_results = self._run_agent_dialogue(experiment, agent_controller, workflow_controller)

            # PHASE 2: Execute the actual ML pipeline
            ml_results = self._execute_ml_pipeline(experiment, config_dict)

            # Combine results
            combined_results = {
                "agent_dialogue": agent_results,
                "ml_pipeline": ml_results
            }

            # Store the combined results
            experiment.results = combined_results

            # Update final status
            experiment.update_status("completed", 1.0, "completed", "")
            experiment.add_log_entry("Experiment completed successfully", level="INFO")

        except Exception as e:
            experiment.update_status("failed")
            experiment.add_log_entry(f"Error running experiment: {str(e)}", level="ERROR")
            import traceback
            experiment.add_log_entry(traceback.format_exc(), level="ERROR")

    def _run_agent_dialogue(self, experiment, agent_controller, workflow_controller):
        """Run the agent dialogue part with direct Mistral API calls"""
        try:
            experiment.update_status("running", 0.2, "agent_dialogue", "")
            experiment.add_log_entry("Running agent dialogue", level="INFO")

            # Get LLM API URL
            llm_api_url = self.llm_api_url

            # Create our direct agent runner
            agent_runner = AgentRunner(llm_api_url)

            # Get the agents and tasks from the workflow
            workflow = workflow_controller.workflows.get(experiment.workflow_id)
            if not workflow:
                experiment.add_log_entry(f"Workflow {experiment.workflow_id} not found", level="ERROR")
                return {"error": "Workflow not found"}

            # Convert workflow tasks to our format
            workflow_tasks = []
            for task_def in workflow.tasks:
                agent_info = agent_controller.get_agent(task_def.agent_id)
                if agent_info:
                    task_info = {
                        "role": agent_info['role'],
                        "goal": agent_info['goal'],
                        "task": task_def.description,
                        "backstory": agent_info.get('backstory', '')
                    }
                    workflow_tasks.append(task_info)

            if not workflow_tasks:
                experiment.add_log_entry("No tasks defined for the workflow", level="ERROR")
                return {"error": "No tasks defined"}

            # Add data context to the first task
            if workflow_tasks and experiment.data_path:
                # Create a data context string
                data_context = f"The data file is located at: {experiment.data_path}\n"
                if os.path.exists(experiment.data_path):
                    try:
                        if experiment.data_path.endswith('.csv'):
                            # Try to read the first few rows for context
                            df = pd.read_csv(experiment.data_path, nrows=5)
                            data_context += f"Data sample (first 5 rows):\n{df.to_string()}\n\n"
                            data_context += f"Data columns: {', '.join(df.columns)}\n"
                            data_context += f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
                    except Exception as e:
                        data_context += f"Could not read data file: {str(e)}\n"

                workflow_tasks[0]['task'] = data_context + "\n" + workflow_tasks[0]['task']

            # Run each task and update progress
            agent_results = {}
            for i, task_info in enumerate(workflow_tasks):
                agent_role = task_info['role']
                progress = 0.2 + (i / len(workflow_tasks) * 0.3)  # Use 30% of progress bar for dialogue
                experiment.update_status("running", progress, f"agent_dialogue_{i + 1}", agent_role)
                experiment.add_log_entry(f"Running agent dialogue {i + 1}: {agent_role}", level="INFO",
                                         agent=agent_role)

                # Run the task
                try:
                    result = agent_runner.run_task(
                        task_info['role'],
                        task_info['goal'],
                        task_info['task'],
                        task_info.get('backstory', '')
                    )
                    agent_results[i] = {
                        "role": agent_role,
                        "task": task_info['task'],
                        "response": result
                    }
                    experiment.add_log_entry(f"Agent dialogue {i + 1} completed", level="INFO", agent=agent_role)
                    experiment.add_log_entry(f"Result: {result[:500]}...", level="INFO", agent=agent_role)
                except Exception as e:
                    experiment.add_log_entry(f"Error in agent dialogue {i + 1}: {str(e)}", level="ERROR",
                                             agent=agent_role)
                    agent_results[i] = {
                        "role": agent_role,
                        "task": task_info['task'],
                        "error": str(e)
                    }

            experiment.add_log_entry("Agent dialogue phase completed", level="INFO")
            return agent_results

        except Exception as e:
            experiment.add_log_entry(f"Error in agent dialogue: {str(e)}", level="ERROR")
            import traceback
            experiment.add_log_entry(traceback.format_exc(), level="ERROR")
            return {"error": str(e)}

    def _execute_ml_pipeline(self, experiment, config_dict):
        """Execute the actual ML pipeline with real anomaly detection logic"""
        try:
            experiment.update_status("running", 0.5, "ml_pipeline", "")
            experiment.add_log_entry("Starting ML pipeline execution", level="INFO")

            # Initialize result dictionary
            results = {
                "data_understanding": None,
                "data_preprocessing": None,
                "feature_engineering": None,
                "data_splitting": None,
                "model_optimization": None,
                "model_training": None,
                "model_evaluation": None,
                "feature_analysis": None,
                "quality_assessment": None
            }

            # Initialize state variables for ML pipeline
            raw_data = None
            processed_data = None
            label_encoders = {}
            features = None
            labels = None
            vectorizer = None
            mlb = None
            feature_names = []
            X_train = None
            X_test = None
            y_train = None
            y_test = None
            best_params = None
            model = None
            evaluation_results = None
            feature_importance = None

            # Step 1: Load and analyze data
            experiment.update_status("running", 0.55, "data_loading", "Data Understanding Specialist")
            experiment.add_log_entry("Loading and analyzing data", level="INFO", agent="Data Understanding Specialist")

            try:
                # Load data
                if experiment.data_path.endswith('.csv'):
                    raw_data = pd.read_csv(experiment.data_path)
                elif experiment.data_path.endswith('.xlsx') or experiment.data_path.endswith('.xls'):
                    raw_data = pd.read_excel(experiment.data_path)
                else:
                    raise ValueError(f"Unsupported file format: {experiment.data_path}")

                # Basic data analysis
                data_analysis = {
                    "shape": raw_data.shape,
                    "columns": raw_data.columns.tolist(),
                    "missing_values": raw_data.isnull().sum().to_dict(),
                    "data_types": {col: str(dtype) for col, dtype in raw_data.dtypes.items()}
                }

                if 'IS_ANOMALY' in raw_data.columns:
                    anomaly_count = raw_data['IS_ANOMALY'].sum()
                    anomaly_ratio = anomaly_count / len(raw_data)
                    data_analysis["anomaly_count"] = int(anomaly_count)
                    data_analysis["anomaly_ratio"] = float(anomaly_ratio)
                    experiment.add_log_entry(
                        f"Anomaly ratio: {anomaly_ratio:.4f} ({anomaly_count} anomalies in {len(raw_data)} records)",
                        level="INFO", agent="Data Understanding Specialist")

                results["data_understanding"] = data_analysis
                experiment.add_log_entry(f"Data loaded with shape: {raw_data.shape}", level="INFO",
                                         agent="Data Understanding Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error loading data: {str(e)}", level="ERROR")
                results["data_understanding"] = {"error": str(e)}
                return results

            # Step 2: Process data
            experiment.update_status("running", 0.6, "data_preprocessing", "Data Preprocessing Engineer")
            experiment.add_log_entry("Processing data", level="INFO", agent="Data Preprocessing Engineer")

            try:
                processor = DataProcessor(config_dict)
                processed_data, label_encoders = processor.process_data(raw_data)

                results["data_preprocessing"] = {
                    "processed_shape": processed_data.shape,
                    "new_columns": [col for col in processed_data.columns if col not in raw_data.columns],
                    "encoded_columns": [col for col in processed_data.columns if col.endswith('_encoded')],
                    "anomaly_columns": [col for col in processed_data.columns if '_anomaly_' in col]
                }

                experiment.add_log_entry(f"Data processed with shape: {processed_data.shape}", level="INFO",
                                         agent="Data Preprocessing Engineer")
            except Exception as e:
                experiment.add_log_entry(f"Error processing data: {str(e)}", level="ERROR")
                results["data_preprocessing"] = {"error": str(e)}
                return results

            # Step 3: Feature engineering
            experiment.update_status("running", 0.65, "feature_engineering", "Feature Engineering Specialist")
            experiment.add_log_entry("Engineering features", level="INFO", agent="Feature Engineering Specialist")

            try:
                engineer = FeatureEngineer(config_dict)
                vectorizer, mlb, features, feature_stats = engineer.create_features(processed_data)

                # Create feature names for later interpretation
                feature_names = []
                if vectorizer:
                    vocab = vectorizer.get_feature_names_out()
                    feature_names.extend([f"tfidf_{word}" for word in vocab])

                for col in config_dict["categorical_columns"]:
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

                # Get labels
                if 'IS_ANOMALY' in processed_data.columns:
                    labels = processed_data['IS_ANOMALY'].values
                else:
                    experiment.add_log_entry("Warning: No target variable 'IS_ANOMALY' found in the data",
                                             level="WARNING")
                    labels = np.zeros(features.shape[0])

                results["feature_engineering"] = feature_stats
                experiment.add_log_entry(f"Features created with shape: {features.shape}", level="INFO",
                                         agent="Feature Engineering Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error engineering features: {str(e)}", level="ERROR")
                results["feature_engineering"] = {"error": str(e)}
                return results

            # Step 4: Split data
            experiment.update_status("running", 0.7, "data_splitting", "Data Splitting Specialist")
            experiment.add_log_entry("Splitting data", level="INFO", agent="Data Splitting Specialist")

            try:
                splitter = DataSplitter(config_dict)
                best_ratio, ratio_results = splitter.find_optimal_anomaly_ratio(
                    features, labels, test_size=0.2, ratios=[0.01, 0.05, 0.1, 0.15, 0.2]
                )

                X_train, X_test, y_train, y_test = splitter.custom_train_test_split(
                    features, labels, test_size=0.2, anomaly_ratio=best_ratio
                )

                results["data_splitting"] = {
                    "best_anomaly_ratio": best_ratio,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "train_anomaly_ratio": float(np.mean(y_train)),
                    "test_anomaly_ratio": float(np.mean(y_test))
                }

                experiment.add_log_entry(f"Best anomaly ratio: {best_ratio}", level="INFO",
                                         agent="Data Splitting Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error splitting data: {str(e)}", level="ERROR")
                results["data_splitting"] = {"error": str(e)}
                return results

            # Step 5: Optimize model
            experiment.update_status("running", 0.75, "model_optimization", "Model Optimization Specialist")
            experiment.add_log_entry("Optimizing model parameters", level="INFO", agent="Model Optimization Specialist")

            try:
                optimizer = GAOptimizer(config_dict, X_train, y_train)
                best_params, best_fitness, fitness_history = optimizer.optimize()

                results["model_optimization"] = {
                    "best_params": best_params,
                    "best_fitness": best_fitness
                }

                experiment.add_log_entry(f"Best fitness: {best_fitness:.4f}", level="INFO",
                                         agent="Model Optimization Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error optimizing model: {str(e)}", level="ERROR")
                results["model_optimization"] = {"error": str(e)}
                return results

            # Step 6: Train model
            experiment.update_status("running", 0.8, "model_training", "Model Training Specialist")
            experiment.add_log_entry("Training model", level="INFO", agent="Model Training Specialist")

            try:
                trainer = ModelTrainer(config_dict)
                model = trainer.train_model(X_train, y_train, best_params, X_test, y_test)

                results["model_training"] = {
                    "model_type": "XGBoost",
                    "num_features": features.shape[1],
                    "best_iteration": model.best_iteration,
                    "best_score": model.best_score
                }

                experiment.add_log_entry(f"Model trained, best score: {model.best_score:.4f}",
                                         level="INFO", agent="Model Training Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error training model: {str(e)}", level="ERROR")
                results["model_training"] = {"error": str(e)}
                return results

            # Step 7: Evaluate model
            experiment.update_status("running", 0.85, "model_evaluation", "Model Evaluation Specialist")
            experiment.add_log_entry("Evaluating model", level="INFO", agent="Model Evaluation Specialist")

            try:
                evaluator = ModelEvaluator(config_dict)
                evaluation_results = evaluator.evaluate_model(model, X_test, y_test)

                # Find optimal threshold
                threshold_results = evaluator.find_optimal_threshold(model, X_test, y_test)

                # Add threshold results to evaluation results
                evaluation_results['optimal_thresholds'] = threshold_results

                results["model_evaluation"] = {
                    "roc_auc": evaluation_results['roc_auc'],
                    "pr_auc": evaluation_results['pr_auc'],
                    "anomaly_precision": evaluation_results['class_1']['precision'],
                    "anomaly_recall": evaluation_results['class_1']['recall'],
                    "anomaly_f1": evaluation_results['class_1']['f1'],
                    "optimal_threshold_f1": evaluation_results['optimal_thresholds']['f1_optimal'],
                    "optimal_threshold_f2": evaluation_results['optimal_thresholds']['f2_optimal']
                }

                experiment.add_log_entry(
                    f"ROC-AUC: {evaluation_results['roc_auc']:.4f}, PR-AUC: {evaluation_results['pr_auc']:.4f}",
                    level="INFO", agent="Model Evaluation Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error evaluating model: {str(e)}", level="ERROR")
                results["model_evaluation"] = {"error": str(e)}
                return results

            # Step 8: Analyze feature importance
            experiment.update_status("running", 0.9, "feature_analysis", "Feature Analysis Specialist")
            experiment.add_log_entry("Analyzing feature importance", level="INFO", agent="Feature Analysis Specialist")

            try:
                feature_importance = engineer.analyze_feature_importance(model, feature_names, top_n=20)
                feature_suggestions = engineer.suggest_features(processed_data, feature_importance)

                results["feature_analysis"] = {
                    "top_features": feature_importance.to_dict() if hasattr(feature_importance, 'to_dict') else None,
                    "suggestions": feature_suggestions
                }

                experiment.add_log_entry("Feature importance analysis completed", level="INFO",
                                         agent="Feature Analysis Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error analyzing feature importance: {str(e)}", level="ERROR")
                results["feature_analysis"] = {"error": str(e)}
                return results

            # Step 9: Quality assessment
            experiment.update_status("running", 0.95, "quality_assessment", "Quality Assessment Specialist")
            experiment.add_log_entry("Assessing model quality", level="INFO", agent="Quality Assessment Specialist")

            try:
                interpretation = evaluator.interpret_results(evaluation_results)

                results["quality_assessment"] = interpretation

                experiment.add_log_entry(f"Quality assessment: {interpretation['summary']}", level="INFO",
                                         agent="Quality Assessment Specialist")
            except Exception as e:
                experiment.add_log_entry(f"Error in quality assessment: {str(e)}", level="ERROR")
                results["quality_assessment"] = {"error": str(e)}
                return results

            # Save model artifacts
            try:
                experiment.add_log_entry("Saving model artifacts", level="INFO")

                metrics = {
                    'roc_auc': evaluation_results['roc_auc'],
                    'pr_auc': evaluation_results['pr_auc'],
                    'optimal_threshold_f1': evaluation_results['optimal_thresholds']['f1_optimal'],
                    'optimal_threshold_f2': evaluation_results['optimal_thresholds']['f2_optimal'],
                    'anomaly_precision': evaluation_results['class_1']['precision'],
                    'anomaly_recall': evaluation_results['class_1']['recall'],
                    'anomaly_f1': evaluation_results['class_1']['f1'],
                    'classification_report': evaluation_results['classification_report'],
                    'confusion_matrix': evaluation_results['confusion_matrix'],
                    'interpretation': interpretation
                }

                artifact_paths = save_model_artifacts(
                    model, vectorizer, mlb, config_dict,
                    label_encoders, metrics, experiment.output_path
                )

                results["artifact_paths"] = artifact_paths
                experiment.add_log_entry("Model artifacts saved", level="INFO")
            except Exception as e:
                experiment.add_log_entry(f"Error saving artifacts: {str(e)}", level="ERROR")
                results["artifact_paths"] = {"error": str(e)}

            experiment.add_log_entry("ML pipeline execution completed", level="INFO")
            return results

        except Exception as e:
            experiment.add_log_entry(f"Error in ML pipeline: {str(e)}", level="ERROR")
            import traceback
            experiment.add_log_entry(traceback.format_exc(), level="ERROR")

    def cancel_experiment(self, experiment_id):
        """Cancel a running experiment."""
        if experiment_id in self.experiments and experiment_id in self.active_runs:
            experiment = self.experiments[experiment_id]
            # In a real implementation, we would need a proper way to cancel the thread
            # For now, we'll just update the status
            experiment.update_status("cancelled")
            experiment.add_log_entry("Experiment cancelled by user", level="INFO")
            return experiment.to_dict()
        return None

    def delete_experiment(self, experiment_id):
        """Delete an experiment."""
        if experiment_id in self.experiments:
            # If the experiment is running, cancel it first
            if experiment_id in self.active_runs:
                self.cancel_experiment(experiment_id)

            # Delete the experiment
            del self.experiments[experiment_id]
            return True
        return False