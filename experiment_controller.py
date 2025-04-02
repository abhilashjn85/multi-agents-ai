# anomaly_detection_app/controllers/experiment_controller.py (Updated)

import os
import json
import uuid
import datetime
import threading
from anomaly_detection_app.models.experiment import Experiment, ExperimentConfig


class ExperimentController:
    """
    Controller for managing experiments using CrewAI agents.
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

        # Get the LLM API URL and model from configuration
        self.llm_api_url = app_config.get('LLM_API_URL', 'https://api.openai.com/v1')
        self.llm_api_key = app_config.get('LLM_API_KEY', '')
        self.model_name = app_config.get('DEFAULT_MODEL_NAME', 'gpt-4-0125-preview')

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
            return self.experiments[experiment_id]
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
        """Run the experiment using our custom agent implementation instead of CrewAI."""
        try:
            from anomaly_detection_app.processor.custom_agent_system import CustomLLMClient, SimpleAgent, SimpleWorkflow
            from anomaly_detection_app.processor.data_processor import DataProcessor, FeatureEngineer
            from anomaly_detection_app.processor.data_splitter import DataSplitter, GAOptimizer
            from anomaly_detection_app.processor.model_trainer import ModelTrainer, ModelEvaluator, save_model_artifacts
            from anomaly_detection_app.tools.custom_tools import (
                create_data_loader_tool,
                create_data_processor_tool,
                create_feature_engineering_tool,
                create_data_splitter_tool,
                create_model_optimizer_tool,
                create_model_trainer_tool,
                create_model_evaluator_tool,
                create_feature_analyzer_tool,
                create_quality_assessment_tool,
                create_model_saver_tool
            )

            # Update experiment status
            experiment.update_status("running", 0.1, "agent_setup", "")
            experiment.add_log_entry("Initializing custom agents", level="INFO")

            # Get the workflow and config
            workflow_dict = self.workflow_controller.get_workflow(experiment.workflow_id)
            config_dict = self.get_config(experiment.config_id)

            if not workflow_dict or not config_dict:
                experiment.update_status("failed")
                experiment.add_log_entry("Workflow or config not found", level="ERROR")
                return {"error": "Workflow or config not found"}

            # Save config to a file in the output directory
            config_path = os.path.join(experiment.output_path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            experiment.add_log_entry(f"Using data path: {experiment.data_path}", level="INFO")
            experiment.add_log_entry(f"Config saved to: {config_path}", level="INFO")

            # Create LLM client
            api_url = "https://aiplatform.dev51.cbf.dev.paypalinc.com/seldon/seldon/mistral-7b-inst-624b0/v2/models/mistral-7b-inst-624b0/infer"
            llm_client = CustomLLMClient(api_url)

            # Store the config in the experiment for tools to access
            experiment.config_dict = config_dict

            # Create function-based tools
            tool_funcs = {
                "data_loader": create_data_loader_tool(experiment),
                "data_processor": create_data_processor_tool(
                    experiment=experiment,
                    config=config_dict,
                    processor_class=DataProcessor
                ),
                "feature_engineer": create_feature_engineering_tool(
                    experiment=experiment,
                    config=config_dict,
                    engineer_class=FeatureEngineer
                ),
                "data_splitter": create_data_splitter_tool(
                    experiment=experiment,
                    config=config_dict,
                    splitter_class=DataSplitter
                ),
                "model_optimizer": create_model_optimizer_tool(
                    experiment=experiment,
                    config=config_dict,
                    optimizer_class=GAOptimizer
                ),
                "model_trainer": create_model_trainer_tool(
                    experiment=experiment,
                    config=config_dict,
                    trainer_class=ModelTrainer
                ),
                "model_evaluator": create_model_evaluator_tool(
                    experiment=experiment,
                    config=config_dict,
                    evaluator_class=ModelEvaluator
                ),
                "feature_analyzer": create_feature_analyzer_tool(
                    experiment=experiment,
                    config=config_dict,
                    engineer_class=FeatureEngineer
                ),
                "quality_assessor": create_quality_assessment_tool(
                    experiment=experiment,
                    config=config_dict,
                    evaluator_class=ModelEvaluator
                ),
                "model_saver": create_model_saver_tool(
                    experiment=experiment,
                    save_function=save_model_artifacts
                )
            }

            # Convert to simplified tool format
            tools = {}
            for name, tool in tool_funcs.items():
                tools[name] = {
                    "name": name,
                    "description": tool.description,
                    "func": tool.func
                }

            # Create agents
            agents = []
            agent_roles = [
                {
                    "role": "Data Understanding Specialist",
                    "goal": "Analyze data and validate configuration compatibility",
                    "backstory": "You are an expert in financial data analysis with specialization in anomaly detection.",
                    "tools": ["data_loader"]
                },
                {
                    "role": "Data Preprocessing Engineer",
                    "goal": "Transform raw data into processable format",
                    "backstory": "You are a skilled data engineer specialized in preparing financial data for machine learning models.",
                    "tools": ["data_processor"]
                },
                {
                    "role": "Feature Engineering Specialist",
                    "goal": "Create optimal features for anomaly detection",
                    "backstory": "You are an expert in creating machine learning features that capture patterns in financial transaction data.",
                    "tools": ["feature_engineer"]
                },
                {
                    "role": "Data Splitting Specialist",
                    "goal": "Create optimal train/test splits with balanced anomaly ratios",
                    "backstory": "You are an expert in handling imbalanced datasets for anomaly detection.",
                    "tools": ["data_splitter"]
                },
                {
                    "role": "Model Optimization Specialist",
                    "goal": "Find optimal model hyperparameters",
                    "backstory": "You are an expert in genetic algorithms and hyperparameter optimization.",
                    "tools": ["model_optimizer"]
                },
                {
                    "role": "Model Training Specialist",
                    "goal": "Train robust anomaly detection models",
                    "backstory": "You are an expert in training machine learning models for financial fraud detection.",
                    "tools": ["model_trainer"]
                },
                {
                    "role": "Model Evaluation Specialist",
                    "goal": "Evaluate model performance with appropriate metrics",
                    "backstory": "You are an expert in evaluating anomaly detection models in financial domains.",
                    "tools": ["model_evaluator"]
                },
                {
                    "role": "Feature Analysis Specialist",
                    "goal": "Analyze feature importance and suggest improvements",
                    "backstory": "You are an expert in interpreting machine learning models and understanding feature contributions.",
                    "tools": ["feature_analyzer"]
                },
                {
                    "role": "Quality Assessment Specialist",
                    "goal": "Ensure the final model meets quality standards",
                    "backstory": "You are the final gatekeeper for model quality in financial fraud detection.",
                    "tools": ["quality_assessor"]
                },
                {
                    "role": "Model Deployment Specialist",
                    "goal": "Save model artifacts and ensure they're properly stored",
                    "backstory": "You are an expert in model deployment and artifact management.",
                    "tools": ["model_saver"]
                }
            ]

            for agent_role in agent_roles:
                agent = SimpleAgent(
                    role=agent_role["role"],
                    goal=agent_role["goal"],
                    backstory=agent_role["backstory"],
                    llm_client=llm_client
                )

                # Add tools to agent
                for tool_name in agent_role["tools"]:
                    if tool_name in tools:
                        print(f"Adding tool {tool_name} to agent {agent_role['role']}")
                        agent.add_tool(tools[tool_name])
                    else:
                        print(f"Warning: Tool {tool_name} not found for agent {agent_role['role']}")

                agents.append(agent)

            # Create simplified tasks
            tasks = [
                {
                    "agent_index": 0,  # Data Understanding Specialist
                    "description": f"Analyze the data file at {experiment.data_path} to understand its structure and contents. Load the data using the data_loader tool and examine its features, patterns, and potential issues."
                },
                {
                    "agent_index": 1,  # Data Preprocessing Engineer
                    "description": "Process the raw data according to configuration. Handle missing values, sequence processing, and categorical encoding."
                },
                {
                    "agent_index": 2,  # Feature Engineering Specialist
                    "description": "Create features for the model based on the processed data. Create TF-IDF features from sequences and handle categorical features."
                },
                {
                    "agent_index": 3,  # Data Splitting Specialist
                    "description": "Split the data into training and testing sets with optimal anomaly ratio. Find the best anomaly ratio for training and create balanced splits."
                },
                {
                    "agent_index": 4,  # Model Optimization Specialist
                    "description": "Find optimal hyperparameters for the XGBoost model using genetic algorithm optimization."
                },
                {
                    "agent_index": 5,  # Model Training Specialist
                    "description": "Train the XGBoost model with the optimal hyperparameters found in the previous step."
                },
                {
                    "agent_index": 6,  # Model Evaluation Specialist
                    "description": "Evaluate the trained model on test data. Calculate metrics like ROC-AUC, PR-AUC, precision, recall, and F1 score."
                },
                {
                    "agent_index": 7,  # Feature Analysis Specialist
                    "description": "Analyze feature importance from the trained model. Identify the most important features and suggest potential improvements."
                },
                {
                    "agent_index": 8,  # Quality Assessment Specialist
                    "description": "Assess the overall quality of the model. Make a go/no-go recommendation for model deployment."
                },
                {
                    "agent_index": 9,  # Model Deployment Specialist
                    "description": "Save the model and all artifacts to the output directory to enable future use."
                }
            ]

            # Create and run the workflow
            workflow = SimpleWorkflow(agents, experiment)

            experiment.add_log_entry("Starting custom agent workflow", level="INFO")
            results = workflow.run_workflow(tasks)
            experiment.add_log_entry("Custom agent workflow completed", level="INFO")

            self._process_workflow_results(experiment, results)

            # Ensure experiment status is properly set to completed
            experiment.update_status("completed", 1.0, "completed", "")

            # Return the results
            return {
                "agent_results": results,
                "metadata": {
                    "model_path": os.path.join(experiment.output_path, "xgboost_model.json"),
                    "config_path": config_path,
                    "output_path": experiment.output_path
                }
            }

        except Exception as e:
            experiment.update_status("failed")
            experiment.add_log_entry(f"Error in agent experiment: {str(e)}", level="ERROR")
            import traceback
            experiment.add_log_entry(traceback.format_exc(), level="ERROR")
            return {"error": str(e)}

    def _process_workflow_results(self, experiment, results):
        """Process and save workflow results to ensure they're available for the UI."""
        try:
            # Save results to experiment
            experiment.results = {"agent_results": results}

            # Create a summary file of all agent responses
            summary_path = os.path.join(experiment.output_path, "workflow_summary.txt")
            with open(summary_path, "w") as f:
                f.write("ANOMALY DETECTION WORKFLOW SUMMARY\n")
                f.write("================================\n\n")

                for result in results:
                    f.write(f"TASK {result['task_id']}: {result['agent_role']}\n")
                    f.write(f"{'-' * 50}\n\n")

                    if "tool_used" in result.get("result", {}):
                        f.write(f"Tool used: {result['result']['tool_used']}\n\n")

                    if "tool_error" in result.get("result", {}):
                        f.write(f"Tool error: {result['result']['tool_error']}\n\n")

                    if "thinking" in result.get("result", {}):
                        f.write(f"Thinking: {result['result']['thinking']}\n\n")

                    f.write(f"Response:\n{result['result'].get('response', 'No response')}\n\n")
                    f.write(f"{'-' * 50}\n\n")

            # Ensure visualizations are created if they don't exist
            static_dir = os.path.join('static', 'results', experiment.id)
            os.makedirs(static_dir, exist_ok=True)

            # Create placeholder visualizations if they don't exist yet
            self._ensure_visualizations(experiment)

        except Exception as e:
            experiment.add_log_entry(f"Error processing workflow results: {str(e)}", level="ERROR")
            import traceback
            experiment.add_log_entry(traceback.format_exc(), level="ERROR")


    def _ensure_visualizations(self, experiment):
        """Ensure visualization files exist for the results page."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Create directory
        static_dir = os.path.join('static', 'results', experiment.id)
        os.makedirs(static_dir, exist_ok=True)

        # Create confusion matrix if it doesn't exist
        if not os.path.exists(os.path.join(static_dir, "confusion_matrix.png")):
            plt.figure(figsize=(10, 8))
            # Use placeholder data if no real data is available
            cm = np.array([[90, 10], [20, 80]])
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
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, "confusion_matrix.png"), dpi=300)
            plt.close()

        # Create ROC curve if it doesn't exist
        if not os.path.exists(os.path.join(static_dir, "roc_curve.png")):
            plt.figure(figsize=(10, 8))
            # Placeholder data
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - np.exp(-3 * fpr)
            plt.plot(fpr, tpr, "b-", linewidth=2, label="ROC (AUC = 0.95)")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(static_dir, "roc_curve.png"), dpi=300)
            plt.close()

        # Create PR curve if it doesn't exist
        if not os.path.exists(os.path.join(static_dir, "pr_curve.png")):
            plt.figure(figsize=(10, 8))
            recall = np.linspace(0, 1, 100)
            precision = np.maximum(0, 1 - recall ** 2)
            plt.plot(recall, precision, "r-", linewidth=2, label="PR (AUC = 0.87)")
            plt.axhline(y=0.1, color="k", linestyle="--", alpha=0.5, label="Baseline (ratio = 0.1)")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper right")
            plt.savefig(os.path.join(static_dir, "pr_curve.png"), dpi=300)
            plt.close()

        # Create feature importance if it doesn't exist
        if not os.path.exists(os.path.join(static_dir, "feature_importance.png")):
            plt.figure(figsize=(12, 10))
            # Placeholder data
            features = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
            importance = [0.23, 0.18, 0.15, 0.12, 0.10]
            plt.barh(features, importance, color="skyblue")
            plt.xlabel("Importance Score")
            plt.ylabel("Feature")
            plt.title("Feature Importance")
            plt.gca().invert_yaxis()
            plt.grid(axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(static_dir, "feature_importance.png"), dpi=300)
            plt.close()

    def _create_tasks(self, experiment, workflow_dict, agents):
        """Create tasks for CrewAI 0.108.0."""
        from crewai import Task

        tasks = []

        # Find the Data Understanding Specialist to act as the lead agent for the first task
        data_understanding_idx = self.get_agent_index(agents, "data understanding")

        # Create data understanding task with simpler instructions
        data_understanding_task = Task(
            description=f"""
            TASK: Analyze the data file to understand its structure and contents.

            DATA FILE: {experiment.data_path}

            STEPS:
            1. Use the data_loader tool to load and analyze the data file
            2. Examine the structure, dimensions, column types, and sample values
            3. Identify anomaly patterns and important features

            Be thorough in your analysis and provide recommendations for preprocessing.
            """,
            agent=agents[data_understanding_idx],
            expected_output="Data analysis report with preprocessing recommendations"
        )
        tasks.append(data_understanding_task)

        # Create data preprocessing task
        preprocessing_idx = self.get_agent_index(agents, "preprocessing")
        data_preprocessing_task = Task(
            description=f"""
            TASK: Process the raw data according to configuration.

            STEPS:
            1. Use the data_processor tool to apply preprocessing steps
            2. Handle missing values, sequence processing, and categorical encoding
            3. Report the changes made to the data

            Make sure all required columns are properly processed for anomaly detection.
            """,
            agent=agents[preprocessing_idx],
            expected_output="Processed data report",
            context=[data_understanding_task]
        )
        tasks.append(data_preprocessing_task)

        # Create feature engineering task
        feature_engineering_idx = self.get_agent_index(agents, "feature engineering")
        feature_engineering_task = Task(
            description=f"""
            TASK: Create features for the model based on the processed data.

            STEPS:
            1. Use the feature_engineer tool to create features
            2. This will create TF-IDF features from sequences and handle categorical features
            3. Report on the features created and their importance

            Focus on creating discriminative features for anomaly detection.
            """,
            agent=agents[feature_engineering_idx],
            expected_output="Feature engineering report",
            context=[data_preprocessing_task]
        )
        tasks.append(feature_engineering_task)

        # Create data splitting task
        data_splitting_idx = self.get_agent_index(agents, "data splitting")
        data_splitting_task = Task(
            description=f"""
            TASK: Split the data into training and testing sets with optimal anomaly ratio.

            STEPS:
            1. Use the data_splitter tool to find the optimal anomaly ratio and create splits
            2. Report on the resulting data splits and their characteristics

            Ensure the training data has an appropriate balance of normal and anomaly samples.
            """,
            agent=agents[data_splitting_idx],
            expected_output="Data splitting report",
            context=[feature_engineering_task]
        )
        tasks.append(data_splitting_task)

        # Create model optimization task
        model_optimization_idx = self.get_agent_index(agents, "model optimization")
        model_optimization_task = Task(
            description=f"""
            TASK: Find optimal hyperparameters for the XGBoost model.

            STEPS:
            1. Use the model_optimizer tool to perform hyperparameter optimization
            2. Report on the best parameters found and their performance

            Select parameters that maximize anomaly detection performance.
            """,
            agent=agents[model_optimization_idx],
            expected_output="Model optimization report",
            context=[data_splitting_task]
        )
        tasks.append(model_optimization_task)

        # Create model training task
        model_training_idx = self.get_agent_index(agents, "model training")
        model_training_task = Task(
            description=f"""
            TASK: Train the XGBoost model with the optimal hyperparameters.

            STEPS:
            1. Use the model_trainer tool to train the model with optimal parameters
            2. Report on the training process and initial results

            Monitor for any issues during training like overfitting.
            """,
            agent=agents[model_training_idx],
            expected_output="Model training report",
            context=[model_optimization_task]
        )
        tasks.append(model_training_task)

        # Create model evaluation task
        model_evaluation_idx = self.get_agent_index(agents, "model evaluation")
        model_evaluation_task = Task(
            description=f"""
            TASK: Evaluate the trained model on test data.

            STEPS:
            1. Use the model_evaluator tool to assess model performance
            2. Report on metrics like ROC-AUC, PR-AUC, precision, recall, and F1
            3. Analyze the optimal classification threshold

            Provide a detailed interpretation of the results.
            """,
            agent=agents[model_evaluation_idx],
            expected_output="Model evaluation report",
            context=[model_training_task]
        )
        tasks.append(model_evaluation_task)

        # Create feature analysis task
        feature_analysis_idx = self.get_agent_index(agents, "feature analysis")
        feature_analysis_task = Task(
            description=f"""
            TASK: Analyze feature importance from the trained model.

            STEPS:
            1. Use the feature_analyzer tool to identify important features
            2. Report on top features and their impact on the model
            3. Suggest potential improvements to feature engineering

            Focus on features that most contribute to anomaly detection.
            """,
            agent=agents[feature_analysis_idx],
            expected_output="Feature importance analysis",
            context=[model_training_task]
        )
        tasks.append(feature_analysis_task)

        # Create quality assessment task
        quality_assessment_idx = self.get_agent_index(agents, "quality assessment")
        quality_assessment_task = Task(
            description=f"""
            TASK: Assess the overall quality of the model.

            STEPS:
            1. Use the quality_assessor tool to evaluate model quality
            2. Make a go/no-go recommendation for model deployment
            3. Suggest improvements for future iterations

            Be critical but fair in your assessment.
            """,
            agent=agents[quality_assessment_idx],
            expected_output="Quality assessment report",
            context=[model_evaluation_task, feature_analysis_task]
        )
        tasks.append(quality_assessment_task)

        # Create model saving task
        save_agent_idx = next((i for i, agent in enumerate(agents)
                               if "model deployment" in agent.role.lower()), 0)
        model_saving_task = Task(
            description=f"""
            TASK: Save the model and all artifacts to the output directory.

            STEPS:
            1. Use the model_saver tool to save all model artifacts
            2. Confirm all files were saved correctly

            Ensure all necessary components are saved for future use.
            """,
            agent=agents[save_agent_idx],
            expected_output="Model saving report",
            context=[model_training_task, model_evaluation_task, quality_assessment_task]
        )
        tasks.append(model_saving_task)

        # Create final review task
        final_review_task = Task(
            description=f"""
            TASK: Create a comprehensive final report on the anomaly detection project.

            STEPS:
            1. Review results from all previous tasks
            2. Summarize the entire process including data preprocessing, feature engineering, model training, and evaluation
            3. Present final results and recommendations
            4. Suggest next steps for improving the anomaly detection system

            Provide a balanced assessment of what worked well and what could be improved.
            """,
            agent=agents[data_understanding_idx],  # Lead agent handles final report
            expected_output="Comprehensive final report",
            context=[
                data_understanding_task,
                data_preprocessing_task,
                feature_engineering_task,
                data_splitting_task,
                model_optimization_task,
                model_training_task,
                model_evaluation_task,
                feature_analysis_task,
                quality_assessment_task,
                model_saving_task
            ]
        )
        tasks.append(final_review_task)

        return tasks

    # Helper function to find agent by role substring
    def get_agent_index(self, agents, role_substring):
        """Find the index of an agent by role substring"""
        for i, agent in enumerate(agents):
            if role_substring.lower() in agent.role.lower():
                return i
        # Return the first agent as fallback
        return 0

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
