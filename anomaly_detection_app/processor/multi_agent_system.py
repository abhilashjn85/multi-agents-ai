import os
import pandas as pd
import numpy as np
import time
import json
from crewai import Agent, Task, Crew, Process

from anomaly_detection_app.models.custom_llm_client import get_custom_llm, get_llm_callback
# Import our custom processing components
from anomaly_detection_app.processor.data_processor import DataProcessor, FeatureEngineer
from anomaly_detection_app.processor.data_splitter import DataSplitter, GAOptimizer
from anomaly_detection_app.processor.model_trainer import ModelTrainer, ModelEvaluator, save_model_artifacts


class MultiAgentSystem:
    """
    Implementation of the multi-agent system for anomaly detection
    using CrewAI framework integrated with the Flask application.
    """

    def __init__(self, config_path, data_path, api_url=None):
        self.config_path = config_path
        self.data_path = data_path
        self.api_url = api_url or "https://aiplatform.dev51.cbf.dev.paypalinc.com/seldon/seldon/mistral-7b-inst-624b0/v2/models/mistral-7b-inst-624b0/infer"

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize state variables
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.labels = None
        self.vectorizer = None
        self.mlb = None
        self.feature_names = []
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_params = None
        self.model = None
        self.evaluation_results = None
        self.feature_importance = None

        # Get the custom LLM callback
        self.llm_callback = self._get_llm_callback()
        # Create output folder
        self.output_path = os.path.join("model_output", f"run_{int(time.time())}")
        os.makedirs(self.output_path, exist_ok=True)

    def _get_llm_callback(self):
        """Get the custom LLM callback function"""
        # Get model name from config if available, otherwise use default
        model_name = self.config.get("llm_model_name", "mistral-7b-inst-2252b")

        # Use the API URL if provided, otherwise use the default URL
        api_url = self.api_url

        # Log the LLM configuration
        print(f"Using custom LLM: {model_name}")
        print(f"API URL: {api_url}")

        # Return the custom LLM callback
        return get_llm_callback(api_url=api_url, model_name=model_name)

    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)

    # In multi_agent_system.py
    def create_tasks(self, agents):
        """Create tasks for the multi-agent system"""
        tasks = []

        # Create tasks for each agent
        for agent_role, agent in agents.items():
            task = Task(
                description=f"Perform {agent_role} task for anomaly detection",
                agent=agent,
                expected_output=f"Results from {agent_role}"
            )
            tasks.append(task)

        return tasks

    def create_agents(self, api_url, model_name="mistral-7b-inst-2252b"):
        """Create all agents for the multi-agent system using custom LLM callback"""
        # Get the custom LLM callback
        llm_callback = get_llm_callback(api_url, model_name)

        # Data Understanding Agent
        data_understanding_agent = Agent(
            role="Data Understanding Specialist",
            goal="Analyze data and validate configuration compatibility",
            backstory="You are an expert in financial data analysis with specialization in " +
                      "anomaly detection. You understand the nuances of financial transaction " +
                      "data and can quickly identify potential issues in data quality.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        # Data Preprocessing Agent
        data_preprocessing_agent = Agent(
            role="Data Preprocessing Engineer",
            goal="Transform raw data into processable format",
            backstory="You are a skilled data engineer specialized in preparing financial data " +
                      "for machine learning models. You excel at handling missing values, " +
                      "transforming sequences, and implementing domain-specific rules.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        # Feature Engineering Agent
        feature_engineering_agent = Agent(
            role="Feature Engineering Specialist",
            goal="Create optimal features for anomaly detection",
            backstory="You are an expert in creating machine learning features that capture " +
                      "patterns in financial transaction data. You understand the importance " +
                      "of sequence representations and can craft features that highlight " +
                      "anomalous behavior.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        # Data Splitting Agent
        data_splitting_agent = Agent(
            role="Data Splitting Specialist",
            goal="Create optimal train/test splits with balanced anomaly ratios",
            backstory="You are an expert in handling imbalanced datasets for anomaly detection. " +
                      "You understand the importance of proper data splitting and can find the " +
                      "optimal anomaly ratio for training effective models.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=self.llm_callback  # Use custom LLM callback
        )

        # Model Optimization Agent
        model_optimization_agent = Agent(
            role="Model Optimization Specialist",
            goal="Find optimal model hyperparameters",
            backstory="You are an expert in genetic algorithms and hyperparameter optimization. " +
                      "You can efficiently navigate large parameter spaces to find the best " +
                      "configuration for XGBoost models in anomaly detection.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        # Model Training Agent
        model_training_agent = Agent(
            role="Model Training Specialist",
            goal="Train robust anomaly detection models",
            backstory="You are an expert in training machine learning models for financial " +
                      "fraud detection. You understand the nuances of XGBoost and can ensure " +
                      "models converge optimally without overfitting.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        # Model Evaluation Agent
        model_evaluation_agent = Agent(
            role="Model Evaluation Specialist",
            goal="Evaluate model performance with appropriate metrics",
            backstory="You are an expert in evaluating anomaly detection models in financial " +
                      "domains. You understand the importance of both precision and recall in " +
                      "fraud detection and can interpret complex performance metrics.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        # Feature Analysis Agent
        feature_analysis_agent = Agent(
            role="Feature Analysis Specialist",
            goal="Analyze feature importance and suggest improvements",
            backstory="You are an expert in interpreting machine learning models and understanding " +
                      "feature contributions. You can identify the most important features for " +
                      "anomaly detection and suggest improvements to feature engineering.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        # Quality Assessment Agent
        quality_assessment_agent = Agent(
            role="Quality Assessment Specialist",
            goal="Ensure the final model meets quality standards",
            backstory="You are the final gatekeeper for model quality in financial fraud detection. " +
                      "You have extensive experience in production machine learning systems and " +
                      "can determine if a model is ready for deployment or needs further refinement.",
            verbose=True,
            allow_delegation=True,
            custom_llm_callback=llm_callback  # Use custom LLM callback
        )

        return {
            "data_understanding": data_understanding_agent,
            "data_preprocessing": data_preprocessing_agent,
            "feature_engineering": feature_engineering_agent,
            "data_splitting": data_splitting_agent,
            "model_optimization": model_optimization_agent,
            "model_training": model_training_agent,
            "model_evaluation": model_evaluation_agent,
            "feature_analysis": feature_analysis_agent,
            "quality_assessment": quality_assessment_agent
        }

    def run_workflow(self):
        """Run the multi-agent workflow for anomaly detection"""
        print(f"Starting multi-agent workflow for anomaly detection")
        print(f"Data file: {self.data_path}")
        print(f"Config file: {self.config_path}")
        print(f"Using custom LLM with API URL: {self.api_url}")

        # Phase 1: CrewAI symbolic workflow (communication between agents)
        agents = self.create_agents(self,
                                    self.api_url
                                    or "https://aiplatform.dev51.cbf.dev.paypalinc.com/seldon/seldon/mistral-7b-inst-624b0/v2/models/mistral-7b-inst-624b0/infer"
                                    )
        tasks = self.create_tasks(agents)

        # Create crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )

        # Execute crew tasks (LLM-based agent communication)
        print("\n--- Starting CrewAI workflow ---")
        try:
            crew_result = crew.kickoff()
            print("\n--- CrewAI workflow completed ---")
            print(f"Crew result: {crew_result}")
        except Exception as e:
            print(f"Error in CrewAI workflow: {str(e)}")
            print("Continuing with ML pipeline execution...")
            crew_result = "Error in CrewAI workflow, continuing with ML pipeline"

        # Execute the actual ML pipeline
        # Phase 2: Actually perform the ML workflow with real code
        # The CrewAI workflow is primarily for agent communication and planning
        # Now we execute the actual ML pipeline based on results from Phase 1
        results = self._execute_ml_pipeline()

        # Add crew results to the output
        results["crew_result"] = crew_result

        return results

    def _execute_ml_pipeline(self):
        """Execute the actual ML pipeline based on the plan from agents"""
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

        try:
            # Step 1: Load and analyze data
            print("\n==== Loading and Analyzing Data ====")
            # Load data from data_path
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                self.raw_data = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")

            print(f"Loaded data with shape: {self.raw_data.shape}")

            # Basic data analysis
            data_analysis = {
                "shape": self.raw_data.shape,
                "columns": self.raw_data.columns.tolist(),
                "missing_values": self.raw_data.isnull().sum().to_dict(),
                "data_types": {col: str(dtype) for col, dtype in self.raw_data.dtypes.items()}
            }

            if 'IS_ANOMALY' in self.raw_data.columns:
                anomaly_count = self.raw_data['IS_ANOMALY'].sum()
                anomaly_ratio = anomaly_count / len(self.raw_data)
                data_analysis["anomaly_count"] = int(anomaly_count)
                data_analysis["anomaly_ratio"] = float(anomaly_ratio)
                print(f"Anomaly ratio: {anomaly_ratio:.4f} ({anomaly_count} anomalies in {len(self.raw_data)} records)")

            results["data_understanding"] = data_analysis

            # Step 2: Process data
            print("\n==== Processing Data ====")
            processor = DataProcessor(self.config)
            self.processed_data, self.label_encoders = processor.process_data(self.raw_data)
            print(f"Processed data shape: {self.processed_data.shape}")

            results["data_preprocessing"] = {
                "processed_shape": self.processed_data.shape,
                "new_columns": [col for col in self.processed_data.columns if col not in self.raw_data.columns],
                "encoded_columns": [col for col in self.processed_data.columns if col.endswith('_encoded')],
                "anomaly_columns": [col for col in self.processed_data.columns if '_anomaly_' in col]
            }

            # Step 3: Feature engineering
            print("\n==== Engineering Features ====")
            engineer = FeatureEngineer(self.config)
            self.vectorizer, self.mlb, self.features, feature_stats = engineer.create_features(self.processed_data)

            # Create feature names for later interpretation
            self.feature_names = []
            if self.vectorizer:
                vocab = self.vectorizer.get_feature_names_out()
                self.feature_names.extend([f"tfidf_{word}" for word in vocab])

            for col in self.config["categorical_columns"]:
                if f"{col}_encoded" in self.processed_data.columns:
                    self.feature_names.append(f"cat_{col}")

            if self.mlb and hasattr(self.mlb, 'classes_'):
                for cls in self.mlb.classes_:
                    self.feature_names.append(f"multi_{cls}")

            for col in self.processed_data.columns:
                if "_anomaly_" in col:
                    self.feature_names.append(col)

            # Make sure we have enough feature names
            while len(self.feature_names) < self.features.shape[1]:
                self.feature_names.append(f"feature_{len(self.feature_names)}")

            # Get labels
            if 'IS_ANOMALY' in self.processed_data.columns:
                self.labels = self.processed_data['IS_ANOMALY'].values
            else:
                print("Warning: No target variable 'IS_ANOMALY' found in the data")
                self.labels = np.zeros(self.features.shape[0])

            results["feature_engineering"] = feature_stats

            # Step 4: Split data
            print("\n==== Splitting Data ====")
            splitter = DataSplitter(self.config)
            best_ratio, ratio_results = splitter.find_optimal_anomaly_ratio(
                self.features, self.labels, test_size=0.2, ratios=[0.01, 0.05, 0.1, 0.15, 0.2]
            )

            self.X_train, self.X_test, self.y_train, self.y_test = splitter.custom_train_test_split(
                self.features, self.labels, test_size=0.2, anomaly_ratio=best_ratio
            )

            results["data_splitting"] = {
                "best_anomaly_ratio": best_ratio,
                "train_size": len(self.X_train),
                "test_size": len(self.X_test),
                "train_anomaly_ratio": float(np.mean(self.y_train)),
                "test_anomaly_ratio": float(np.mean(self.y_test)),
                "ratio_comparison": ratio_results.to_dict() if hasattr(ratio_results, 'to_dict') else None
            }

            # Step 5: Optimize model
            print("\n==== Optimizing Model Parameters ====")
            optimizer = GAOptimizer(self.config, self.X_train, self.y_train)
            self.best_params, best_fitness, fitness_history = optimizer.optimize()

            results["model_optimization"] = {
                "best_params": self.best_params,
                "best_fitness": best_fitness,
                "fitness_history": fitness_history
            }

            # Step 6: Train model
            print("\n==== Training Model ====")
            trainer = ModelTrainer(self.config)
            self.model = trainer.train_model(
                self.X_train, self.y_train, self.best_params, self.X_test, self.y_test
            )

            results["model_training"] = {
                "model_type": "XGBoost",
                "num_features": self.features.shape[1],
                "best_iteration": self.model.best_iteration,
                "best_score": self.model.best_score
            }

            # Step 7: Evaluate model
            print("\n==== Evaluating Model ====")
            evaluator = ModelEvaluator(self.config)
            self.evaluation_results = evaluator.evaluate_model(self.model, self.X_test, self.y_test)

            # Find optimal threshold
            threshold_results = evaluator.find_optimal_threshold(self.model, self.X_test, self.y_test)

            # Add threshold results to evaluation results
            self.evaluation_results['optimal_thresholds'] = threshold_results

            results["model_evaluation"] = self.evaluation_results

            # Step 8: Analyze feature importance
            print("\n==== Analyzing Feature Importance ====")
            self.feature_importance = engineer.analyze_feature_importance(
                self.model, self.feature_names, top_n=20
            )

            feature_suggestions = engineer.suggest_features(
                self.processed_data, self.feature_importance
            )

            results["feature_analysis"] = {
                "top_features": self.feature_importance.to_dict() if hasattr(self.feature_importance,
                                                                             'to_dict') else None,
                "suggestions": feature_suggestions
            }

            # Step 9: Quality assessment
            print("\n==== Quality Assessment ====")
            interpretation = evaluator.interpret_results(self.evaluation_results)

            results["quality_assessment"] = interpretation

            # Save model artifacts
            print("\n==== Saving Model Artifacts ====")
            metrics = {
                'roc_auc': self.evaluation_results['roc_auc'],
                'pr_auc': self.evaluation_results['pr_auc'],
                'optimal_threshold_f1': self.evaluation_results['optimal_thresholds']['f1_optimal'],
                'optimal_threshold_f2': self.evaluation_results['optimal_thresholds']['f2_optimal'],
                'anomaly_precision': self.evaluation_results['class_1']['precision'],
                'anomaly_recall': self.evaluation_results['class_1']['recall'],
                'anomaly_f1': self.evaluation_results['class_1']['f1'],
                'classification_report': self.evaluation_results['classification_report'],
                'confusion_matrix': self.evaluation_results['confusion_matrix'],
                'interpretation': interpretation
            }

            artifact_paths = save_model_artifacts(
                self.model, self.vectorizer, self.mlb, self.config,
                self.label_encoders, metrics, self.output_path
            )

            results["artifact_paths"] = artifact_paths

            # Save overall results
            with open(os.path.join(self.output_path, 'results.json'), 'w') as f:
                json.dump(self._make_json_serializable(results), f, indent=2)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in ML pipeline: {str(e)}")
            print(error_trace)

            # Record the error
            results["error"] = {
                "message": str(e),
                "traceback": error_trace
            }

        return results

    def _make_json_serializable(self, obj):
        """Convert values to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        else:
            return obj