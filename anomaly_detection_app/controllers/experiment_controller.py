# anomaly_detection_app/controllers/experiment_controller.py (Updated)

import os
import json
import types
import uuid
import datetime
import threading
import time
import pandas as pd
import numpy as np
from anomaly_detection_app.models.experiment import Experiment, ExperimentConfig
from anomaly_detection_app.models.workflow import Workflow
from anomaly_detection_app.processor.data_processor import DataProcessor, FeatureEngineer
from anomaly_detection_app.processor.data_splitter import DataSplitter, GAOptimizer
from anomaly_detection_app.processor.model_trainer import ModelTrainer, ModelEvaluator, save_model_artifacts
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool, tool
from langchain.llms import OpenAI
from pydantic import ConfigDict
from typing import ClassVar, Any

from anomaly_detection_app.tools.custom_tools import ModelSaverTool, QualityAssessmentTool, FeatureAnalyzerTool, \
    ModelOptimizerTool, ModelTrainerTool, ModelEvaluatorTool, DataSplitterTool, FeatureEngineeringTool, \
    DataProcessorTool, DataLoaderTool


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

            # Check if workflow and config exist
            if not self.workflow_controller.get_workflow(workflow_id) or not self.get_config(config_id):
                experiment.update_status("failed")
                experiment.add_log_entry("Workflow or config not found", level="ERROR")
                return

            # Run the experiment using CrewAI agents
            experiment.add_log_entry("Starting the experiment using CrewAI agents", level="INFO")
            results = self._run_agent_experiment(experiment)

            # Store the results
            experiment.results = results

            # Update final status
            experiment.update_status("completed", 1.0, "completed", "")
            experiment.add_log_entry("Experiment completed successfully", level="INFO")

        except Exception as e:
            experiment.update_status("failed")
            experiment.add_log_entry(
                f"Error running experiment: {str(e)}", level="ERROR"
            )
            import traceback
            experiment.add_log_entry(traceback.format_exc(), level="ERROR")

    def _run_agent_experiment(self, experiment):
        """
        Run the experiment using CrewAI agents that handle both planning and execution.
        """
        try:
            # Update experiment status
            experiment.update_status("running", 0.1, "agent_setup", "")
            experiment.add_log_entry("Initializing CrewAI agents", level="INFO")

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
            # Control CrewAI behavior
            os.environ["CREWAI_TIMEOUT"] = "30"
            os.environ["CREWAI_MAX_TOOL_ATTEMPTS"] = "2"
            os.environ["CREWAI_STRICT_FORMAT"] = "False"

            tools = self._create_tools(experiment, config_dict)

            # Create agents based on the workflow
            agents = self._create_agents(experiment, workflow_dict, tools)

            # Set all agents to not allow delegation if it's causing problems
            for agent in agents:
                agent.allow_delegation = False

            # Create tasks based on the workflow
            tasks = self._create_tasks(experiment, workflow_dict, agents)

            for i in range(1, len(tasks)):
                prev_task = tasks[i - 1]
                if prev_task not in tasks[i].context:
                    tasks[i].context.append(prev_task)

            # Create and run the crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )

            # Run the crew
            experiment.add_log_entry("Starting CrewAI execution", level="INFO")
            results = crew.kickoff()

            # Return the results
            experiment.add_log_entry("CrewAI execution completed", level="INFO")
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

    def _create_tools(self, experiment, config_dict):
        """Create tools for the agents based on ML components"""
        from anomaly_detection_app.processor.data_processor import DataProcessor, FeatureEngineer
        from anomaly_detection_app.processor.data_splitter import DataSplitter, GAOptimizer
        from anomaly_detection_app.processor.model_trainer import ModelTrainer, ModelEvaluator, save_model_artifacts

        # Store the config in the experiment for tools to access
        experiment.config_dict = config_dict

        # Create tool instances
        return {
            "data_loader": DataLoaderTool(experiment=experiment),
            "data_processor": DataProcessorTool(
                experiment=experiment,
                config=config_dict,
                processor_class=DataProcessor
            ),
            "feature_engineer": FeatureEngineeringTool(
                experiment=experiment,
                config=config_dict,
                engineer_class=FeatureEngineer
            ),
            "data_splitter": DataSplitterTool(
                experiment=experiment,
                config=config_dict,
                splitter_class=DataSplitter
            ),
            "model_optimizer": ModelOptimizerTool(
                experiment=experiment,
                config=config_dict,
                optimizer_class=GAOptimizer
            ),
            "model_trainer": ModelTrainerTool(
                experiment=experiment,
                config=config_dict,
                trainer_class=ModelTrainer
            ),
            "model_evaluator": ModelEvaluatorTool(
                experiment=experiment,
                config=config_dict,
                evaluator_class=ModelEvaluator
            ),
            "feature_analyzer": FeatureAnalyzerTool(
                experiment=experiment,
                config=config_dict,
                engineer_class=FeatureEngineer
            ),
            "quality_assessor": QualityAssessmentTool(
                experiment=experiment,
                config=config_dict,
                evaluator_class=ModelEvaluator
            ),
            "model_saver": ModelSaverTool(
                experiment=experiment,
                save_function=save_model_artifacts
            )
        }

    def _create_agents(self, experiment, workflow_dict, tools):
        """Create agents based on workflow"""
        agents = []
        agent_mapping = {}
        model_name = "mistral-7b-inst-624b0"
        host_url = "aiplatform.dev51.cbf.dev.paypalinc.com"
        # Modify OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "EMPTY"
        openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        os.environ["CREWAI_TIMEOUT"] = "30"
        os.environ["CREWAI_MAX_TOOL_ATTEMPTS"] = "2"
        os.environ["CREWAI_STRICT_FORMAT"] = "False"
        openai_api_base = (
                "https://"
                + host_url
                + "/seldon/seldon/"
                + model_name
                + "/v2/models/"
                + model_name
        )
        llm = LLM(
            model="openai/" + model_name,
            api_key=openai_api_key,
            base_url=openai_api_base
        )

        # Add manager agent
        manager_agent = Agent(
            role="Senior Data Scientist",
            goal="Oversee and coordinate the anomaly detection process",
            backstory="You are an experienced data science team lead with expertise in anomaly detection. "
                      "You coordinate specialists and make high-level decisions about the model development process.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        agents.append(manager_agent)
        agent_mapping["manager"] = 0

        # Add specialized agents based on workflow
        for agent_id in workflow_dict.get('agent_ids', []):
            agent_info = self.agent_controller.get_agent(agent_id)
            if agent_info:
                agent_tools = []

                # Assign tools based on agent role
                if "Data Understanding" in agent_info['role']:
                    agent_tools = [tools["data_loader"]]
                elif "Preprocessing" in agent_info['role']:
                    agent_tools = [tools["data_processor"]]
                elif "Feature Engineering" in agent_info['role']:
                    agent_tools = [tools["feature_engineer"]]
                elif "Data Splitting" in agent_info['role']:
                    agent_tools = [tools["data_splitter"]]
                elif "Model Optimization" in agent_info['role']:
                    agent_tools = [tools["model_optimizer"]]
                elif "Model Training" in agent_info['role']:
                    agent_tools = [tools["model_trainer"]]
                elif "Model Evaluation" in agent_info['role']:
                    agent_tools = [tools["model_evaluator"]]
                elif "Feature Analysis" in agent_info['role']:
                    agent_tools = [tools["feature_analyzer"]]
                elif "Quality Assessment" in agent_info['role']:
                    agent_tools = [tools["quality_assessor"]]

                agent = Agent(
                    role=agent_info['role'],
                    goal=agent_info['goal'],
                    backstory=agent_info.get('backstory', ''),
                    verbose=agent_info.get('verbose', False),
                    allow_delegation=agent_info.get('allow_delegation', False),
                    llm=llm,
                    tools=agent_tools
                )
                agents.append(agent)
                agent_mapping[agent_id] = len(agents) - 1  # Store index in agents list

                # Ensure we have the agent for saving artifacts
            if not any("Model Saver" in agent.role for agent in agents):
                save_agent = Agent(
                    role="Model Deployment Specialist",
                    goal="Save model artifacts and ensure they're properly stored",
                    backstory="You are an expert in model deployment and artifact management. Your "
                              "responsibility is to ensure all model components are properly saved and documented.",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm,
                    tools=[tools["model_saver"]]
                )
                agents.append(save_agent)
                agent_mapping["model_saver"] = len(agents) - 1

                # Update experiment with agent mapping for logging purposes
            experiment.agent_mapping = agent_mapping

        return agents

    def _create_tasks(self, experiment, workflow_dict, agents):
        """Create tasks based on workflow without a manager agent"""
        tasks = []

        # Find the Data Understanding Specialist to act as the lead agent for the first task
        lead_agent_index = next((i for i, agent in enumerate(agents)
                                 if "data understanding" in agent.role.lower()), 0)

        # Create an initial planning task
        planning_task = Task(
            description=f"""
            Create a detailed execution plan for anomaly detection on financial transaction data.
            The data is located at {experiment.data_path}.

            Your plan should include:
            1. Data understanding and preparation approach
            2. Feature engineering strategy
            3. Model selection and hyperparameter optimization
            4. Evaluation criteria and quality assessment

            This plan will guide the entire anomaly detection process.
            
            TOOL INFORMATION:
            - Tool name: data_loader
            - Required parameter: data_path (string)
    
            Example usage:
            Action: data_loader
            Action Input: {{"data_path": "{experiment.data_path}"}}
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available.
    
            """,
            agent=agents[lead_agent_index],
            expected_output="A comprehensive execution plan for anomaly detection"
        )
        tasks.append(planning_task)

        # Create data understanding task (continuation of planning)
        data_understanding_task = Task(
            description=f"""
            Create a detailed execution plan for anomaly detection on financial transaction data.
    The data is located at {experiment.data_path}.
    
    Your plan should include:
    1. Data understanding and preparation approach
    2. Feature engineering strategy
    3. Model selection and hyperparameter optimization
    4. Evaluation criteria and quality assessment
    
    This plan will guide the entire anomaly detection process.
    
    TOOL INFORMATION:
    - You can use data_loader to load the data
    - Required parameter: data_path (string)
    
    IMPORTANT FORMAT INSTRUCTIONS:
    - ONLY use the data_loader tool once to get the data
    - After loading the data, provide your final answer
    - Do NOT try to use the tool again after you've used it once
    - Follow EXACTLY this format:
    
    Thought: [your thinking process]
    Action: data_loader
    Action Input: {{"data_path": "{experiment.data_path}"}}
    
    Then after seeing the results:
    
    Thought: I now have the data and can create the execution plan
    Final Answer: [your complete execution plan]
    """,
            agent=agents[self.get_agent_index(agents, "data understanding")],
            expected_output="Data analysis report with preprocessing recommendations",
            context=[planning_task]
        )
        tasks.append(data_understanding_task)

        # Create data preprocessing task
        data_preprocessing_task = Task(
            description=f"""
            Process the raw data according to configuration based on the data understanding report.

            Your task:
            1. Apply the configuration rules to the data
            2. Handle missing values
            3. Process sequences
            4. Report on the preprocessing results

            IMPORTANT INSTRUCTIONS:
            - You have EXACTLY ONE tool available: "data_processor"
            - You MUST use the tool exactly as shown below:
    
            Action: data_processor
            Action Input: {{}}
    
            - Do NOT try to use any other tools
            - Do NOT change the tool name
            - Do NOT add parameters to the tool name
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available.
            """,
            agent=agents[self.get_agent_index(agents, "preprocessing")],
            expected_output="Processed data report",
            context=[data_understanding_task]
        )
        tasks.append(data_preprocessing_task)

        # Create feature engineering task
        feature_engineering_task = Task(
            description=f"""
            Create features for the model based on the processed data.

            Your task:
            1. Create TF-IDF features from sequences
            2. Handle categorical features
            3. Create features from anomaly rules
            4. Report on the feature engineering results
            
            TOOL INFORMATION:
            - Tool name: feature_engineer
            - No parameters needed
            
            Example usage:
            Action: feature_engineer
            Action Input: {{}}
            
            Note: If you encounter errors after a few attempts, please provide your best analysis with the information available.
            Do NOT modify the tool name by adding parameters to it like "feature_engineer(sequence)". 
            The tool name must be exactly "feature_engineer" and the input must be an empty JSON object.
            """,
            agent=agents[self.get_agent_index(agents, "feature engineering")],
            expected_output="Feature engineering report",
            context=[data_preprocessing_task]
        )
        tasks.append(feature_engineering_task)

        # Create data splitting task
        data_splitting_task = Task(
            description=f"""
            Split the data into training and testing sets with optimal anomaly ratio.

            Your task:
            1. Find the optimal anomaly ratio for training
            2. Create balanced train/test splits
            3. Report on the data splitting statistics

            TOOL INFORMATION:
            - Tool name: data_splitter
            - No additional parameters needed
    
            Example usage:
            Action: data_splitter
            Action Input: {{}}
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available.
            """,
            agent=agents[self.get_agent_index(agents, "data splitting")],
            expected_output="Data splitting report",
            context=[feature_engineering_task]
        )
        tasks.append(data_splitting_task)

        # Create model optimization task
        model_optimization_task = Task(
            description=f"""
            Find optimal hyperparameters for the XGBoost model using genetic algorithm.

            Your task:
            1. Run genetic algorithm optimization
            2. Evaluate different parameter combinations
            3. Report on the best parameters found

            TOOL INFORMATION:
            - Tool name: model_optimizer
            - No additional parameters needed
    
            Example usage:
            Action: model_optimizer
            Action Input: {{}}
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available.
            """,
            agent=agents[self.get_agent_index(agents, "model optimization")],
            expected_output="Model optimization report",
            context=[data_splitting_task]
        )
        tasks.append(model_optimization_task)

        # Create model training task
        model_training_task = Task(
            description=f"""
            Train the XGBoost model with the optimal hyperparameters.

            Your task:
            1. Train model with the best parameters
            2. Monitor training progress
            3. Report on training results

            TOOL INFORMATION:
            - Tool name: model_trainer
            - No additional parameters needed
    
            Example usage:
            Action: model_trainer
            Action Input: {{}}
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available. 
            """,
            agent=agents[self.get_agent_index(agents, "model training")],
            expected_output="Model training report",
            context=[model_optimization_task]
        )
        tasks.append(model_training_task)

        # Create model evaluation task
        model_evaluation_task = Task(
            description=f"""
            Evaluate the trained model on test data.

            Your task:
            1. Calculate ROC-AUC, PR-AUC, precision, recall, and F1
            2. Find optimal classification threshold
            3. Generate confusion matrix
            4. Report on model performance

            TOOL INFORMATION:
            - Tool name: model_evaluator
            - No additional parameters needed
    
            Example usage:
            Action: model_evaluator
            Action Input: {{}}
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available. 
            """,
            agent=agents[self.get_agent_index(agents, "model evaluation")],
            expected_output="Model evaluation report",
            context=[model_training_task]
        )
        tasks.append(model_evaluation_task)

        # Create feature analysis task
        feature_analysis_task = Task(
            description=f"""
            Analyze feature importance from the trained model.

            Your task:
            1. Identify the most important features
            2. Analyze top features and their impact
            3. Suggest feature improvements

            TOOL INFORMATION:
            - Tool name: feature_analyzer
            - No additional parameters needed
    
            Example usage:
            Action: feature_analyzer
            Action Input: {{}}
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available. 
            """,
            agent=agents[self.get_agent_index(agents, "feature analysis")],
            expected_output="Feature importance analysis",
            context=[model_training_task]
        )
        tasks.append(feature_analysis_task)

        # Create quality assessment task
        quality_assessment_task = Task(
            description=f"""
            Assess the overall quality of the model.

            Your task:
            1. Interpret evaluation metrics
            2. Identify strengths and weaknesses
            3. Make a go/no-go recommendation
            4. Suggest improvements if needed

            TOOL INFORMATION:
            - Tool name: quality_assessment
            - No additional parameters needed
    
            Example usage:
            Action: quality_assessment
            Action Input: {{}}
    
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available. 
            """,
            agent=agents[self.get_agent_index(agents, "quality assessment")],
            expected_output="Quality assessment report",
            context=[model_evaluation_task, feature_analysis_task]
        )
        tasks.append(quality_assessment_task)

        # Create model saving task
        save_agent_index = next((i for i, agent in enumerate(agents)
                                 if "model deployment" in agent.role.lower()), 0)
        model_saving_task = Task(
            description=f"""
            Save the model and all artifacts to the output directory.

            Your task:
            1. Save the model, vectorizer, and other artifacts
            2. Save evaluation metrics
            3. Report on saved artifacts

            TOOL INFORMATION:
            - Tool name: model_saver
            - No additional parameters needed
    
            Example usage:
            Action: model_saver
            Action Input: {{}}
            
            You must use the exact tool name "model_saver" and provide an empty JSON object as input.
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available.    
            """,
            agent=agents[save_agent_index],
            expected_output="Model saving report",
            context=[model_training_task, model_evaluation_task, quality_assessment_task]
        )
        tasks.append(model_saving_task)

        # Create final review task (assigned to the lead agent from the first task)
        final_review_task = Task(
            description=f"""
            Create a comprehensive final report on the anomaly detection project.

            Your task:
            1. Summarize the entire process
            2. Highlight key findings from each specialist
            3. Present final results and recommendations
            4. Suggest next steps
            Note: If you encounter errors after 3 attempts, please provide your best analysis with the information available. 
            """,
            agent=agents[lead_agent_index],  # Lead agent handles final report
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
