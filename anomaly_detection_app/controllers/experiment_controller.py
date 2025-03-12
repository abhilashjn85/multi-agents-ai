import os
import json
import uuid
import datetime
import threading
import time
import pandas as pd
import numpy as np
import importlib.util
import sys
from models.experiment import Experiment, ExperimentConfig
from models.workflow import Workflow
from crewai import Crew, Process


class ExperimentController:
    """
    Controller for managing experiments.
    """

    def __init__(self, app_config):
        """Initialize the experiment controller."""
        self.app_config = app_config
        self.experiments = {}
        self.configs = {}
        self.active_runs = {}
        self.data_path = ""
        self.output_path = app_config['MODEL_OUTPUT_FOLDER']

        # Load default configuration
        self._load_default_config()

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

            # Check if workflow and config exist in their respective controllers
            # For a real implementation, this would access the controllers directly
            # For now, we'll simulate the workflow execution

            # Load the original code components dynamically
            self._simulate_experiment_run(experiment)
        except Exception as e:
            experiment.update_status("failed")
            experiment.add_log_entry(f"Error running experiment: {str(e)}", level="ERROR")

    def _simulate_experiment_run(self, experiment):
        """
        Simulate the execution of the experiment using the original code components.
        In a real implementation, this would use the actual CrewAI framework
        and the agent implementations from the original code.
        """
        # Update experiment status
        experiment.update_status("running", 0.1, "data_loading", "")
        experiment.add_log_entry("Loading data and configuration", level="INFO")

        # Sleep to simulate loading time
        time.sleep(2)

        # Update progress
        experiment.update_status("running", 0.2, "data_preprocessing", "Data Preprocessing Engineer")
        experiment.add_log_entry("Preprocessing data", level="INFO", agent="Data Preprocessing Engineer")

        # Sleep to simulate processing time
        time.sleep(3)

        # Update progress
        experiment.update_status("running", 0.4, "feature_engineering", "Feature Engineering Specialist")
        experiment.add_log_entry("Engineering features", level="INFO", agent="Feature Engineering Specialist")

        # Sleep to simulate processing time
        time.sleep(3)

        # Update progress
        experiment.update_status("running", 0.6, "model_optimization", "Model Optimization Specialist")
        experiment.add_log_entry("Optimizing model hyperparameters", level="INFO",
                                 agent="Model Optimization Specialist")

        # Sleep to simulate processing time
        time.sleep(4)

        # Update progress
        experiment.update_status("running", 0.8, "model_training", "Model Training Specialist")
        experiment.add_log_entry("Training model", level="INFO", agent="Model Training Specialist")

        # Sleep to simulate processing time
        time.sleep(3)

        # Update progress
        experiment.update_status("running", 0.9, "model_evaluation", "Model Evaluation Specialist")
        experiment.add_log_entry("Evaluating model", level="INFO", agent="Model Evaluation Specialist")

        # Sleep to simulate processing time
        time.sleep(2)

        # Generate some sample results
        results = {
            "metrics": {
                "roc_auc": 0.95,
                "pr_auc": 0.87,
                "anomaly_precision": 0.92,
                "anomaly_recall": 0.85,
                "f1_score": 0.88,
                "optimal_threshold": 0.35
            },
            "feature_importance": [
                {"feature": "feature_1", "importance": 0.23},
                {"feature": "feature_2", "importance": 0.18},
                {"feature": "feature_3", "importance": 0.15},
                {"feature": "feature_4", "importance": 0.12},
                {"feature": "feature_5", "importance": 0.10}
            ],
            "confusion_matrix": [[985, 15], [5, 95]]
        }

        # Save results to the experiment
        experiment.results = results

        # Update final status
        experiment.update_status("completed", 1.0, "completed", "")
        experiment.add_log_entry("Experiment completed successfully", level="INFO")

        # In a real implementation, here we would actually execute the CrewAI workflow
        # and run the actual anomaly detection code from the original implementation

    def run_actual_experiment(self, experiment, agent_controller, workflow_controller):
        """
        Run the actual experiment using CrewAI and the original code components.
        This method would be used in a real implementation.
        """
        # Get the workflow and config
        workflow_dict = workflow_controller.get_workflow(experiment.workflow_id)
        config_dict = self.get_config(experiment.config_id)

        if not workflow_dict or not config_dict:
            experiment.update_status("failed")
            experiment.add_log_entry("Workflow or config not found", level="ERROR")
            return

        # Get the agents for this workflow
        workflow = Workflow.from_dict(workflow_dict)
        agents = []
        for agent_id in workflow.agent_ids:
            agent_dict = agent_controller.get_agent(agent_id)
            if agent_dict:
                # Convert to CrewAI Agent
                agent_obj = agent_controller.to_crew_agent(agent_dict)
                agents.append(agent_obj)

        # Get the tasks for this workflow
        tasks = []
        for task_dict in workflow.tasks:
            # Convert to CrewAI Task
            task_obj = self._create_crew_task(task_dict, agents)
            tasks.append(task_obj)

        # Create CrewAI Crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential if workflow.process_type == "sequential" else Process.hierarchical
        )

        # Run the crew
        try:
            experiment.update_status("running", 0.1, "initialization", "")
            experiment.add_log_entry("Starting experiment with CrewAI", level="INFO")

            # Execute the crew
            result = crew.kickoff()

            # Process results
            experiment.results = result
            experiment.update_status("completed", 1.0, "completed", "")
            experiment.add_log_entry("Experiment completed successfully", level="INFO")
        except Exception as e:
            experiment.update_status("failed")
            experiment.add_log_entry(f"Error running experiment: {str(e)}", level="ERROR")

    def _create_crew_task(self, task_dict, agents):
        """
        Create a CrewAI Task from a task dictionary.
        This is a placeholder for the real implementation.
        """
        # In a real implementation, this would create a CrewAI Task
        # For now, just return the dictionary
        return task_dict

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