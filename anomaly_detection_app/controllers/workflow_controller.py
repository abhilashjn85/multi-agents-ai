import os
import json
import uuid
import datetime
from models.workflow import Workflow, Connection, TaskDefinition


class WorkflowController:
    """
    Controller for managing workflows.
    """

    def __init__(self):
        """Initialize the workflow controller."""
        self.workflows = {}
        self._load_default_workflows()

    def _load_default_workflows(self):
        """Load the default workflow based on the original code."""
        # Create a default workflow for anomaly detection
        default_workflow = Workflow(
            id=str(uuid.uuid4()),
            name="Anomaly Detection Workflow",
            description="Default workflow for anomaly detection using XGBoost and LLM agents",
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
            process_type="sequential"
        )

        # We'll add actual agents later when we have the agent IDs
        self.workflows[default_workflow.id] = default_workflow

    def get_workflows(self):
        """Get all workflows."""
        return [workflow.to_dict() for workflow in self.workflows.values()]

    def get_workflow(self, workflow_id):
        """Get a specific workflow by ID."""
        if workflow_id in self.workflows:
            return self.workflows[workflow_id].to_dict()
        return None

    def create_workflow(self, data):
        """Create a new workflow."""
        # Ensure the workflow has a unique ID
        if 'id' not in data or not data['id']:
            data['id'] = str(uuid.uuid4())

        # Set created and updated timestamps
        data['created_at'] = datetime.datetime.now().isoformat()
        data['updated_at'] = datetime.datetime.now().isoformat()

        workflow = Workflow.from_dict(data)
        self.workflows[workflow.id] = workflow
        return workflow.to_dict()

    def update_workflow(self, workflow_id, data):
        """Update a workflow."""
        if workflow_id in self.workflows:
            # Update the workflow with the new data
            workflow = self.workflows[workflow_id]

            # Handle special collections like connections and tasks
            if 'connections' in data:
                connections = [Connection(**conn) for conn in data.pop('connections')]
                workflow.connections = connections

            if 'tasks' in data:
                tasks = [TaskDefinition(**task) for task in data.pop('tasks')]
                workflow.tasks = tasks

            # Update other attributes
            for key, value in data.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)

            # Update the timestamp
            workflow.updated_at = datetime.datetime.now().isoformat()

            return workflow.to_dict()
        return None

    def delete_workflow(self, workflow_id):
        """Delete a workflow."""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            return True
        return False

    def create_default_anomaly_detection_workflow(self, agent_ids):
        """
        Create a default anomaly detection workflow with the specified agents.
        This method assumes that agent_ids is a dictionary mapping agent names to their IDs.
        """
        # Create a new workflow
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name="Anomaly Detection Workflow",
            description="Default workflow for anomaly detection using XGBoost and LLM agents",
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
            process_type="sequential"
        )

        # Add agents to the workflow
        for agent_id in agent_ids.values():
            workflow.add_agent(agent_id)

        # Define the sequence of agents
        agent_sequence = [
            "Data Understanding Specialist",
            "Data Preprocessing Engineer",
            "Feature Engineering Specialist",
            "Data Splitting Specialist",
            "Model Optimization Specialist",
            "Model Training Specialist",
            "Model Evaluation Specialist",
            "Feature Analysis Specialist",
            "Quality Assessment Specialist"
        ]

        # Add connections between agents
        for i in range(len(agent_sequence) - 1):
            source_name = agent_sequence[i]
            target_name = agent_sequence[i + 1]

            if source_name in agent_ids and target_name in agent_ids:
                source_id = agent_ids[source_name]
                target_id = agent_ids[target_name]
                workflow.add_connection(source_id, target_id)

        # Add tasks for each agent
        # Data Understanding Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Data Understanding Specialist"),
            description="Analyze the data and validate configuration compatibility. " +
                        "Identify any potential issues for anomaly detection.",
            expected_output="A detailed analysis of the data structure and quality issues."
        )

        # Data Preprocessing Engineer
        workflow.add_task(
            agent_id=agent_ids.get("Data Preprocessing Engineer"),
            description="Process the raw data according to the configuration including sequence processing, " +
                        "categorical encoding, and implementing anomaly rules.",
            expected_output="Processed data ready for feature engineering.",
            depends_on=[agent_ids.get("Data Understanding Specialist")]
        )

        # Feature Engineering Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Feature Engineering Specialist"),
            description="Create optimal features for anomaly detection from the processed data including " +
                        "TF-IDF embeddings, categorical features, and anomaly indicators.",
            expected_output="Feature matrix ready for model training.",
            depends_on=[agent_ids.get("Data Preprocessing Engineer")]
        )

        # Data Splitting Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Data Splitting Specialist"),
            description="Find the optimal anomaly ratio for training and create the best train/test split.",
            expected_output="Optimal train/test split with appropriate anomaly ratio.",
            depends_on=[agent_ids.get("Feature Engineering Specialist")]
        )

        # Model Optimization Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Model Optimization Specialist"),
            description="Use genetic algorithm to find optimal hyperparameters for the XGBoost model.",
            expected_output="Optimal hyperparameters for XGBoost.",
            depends_on=[agent_ids.get("Data Splitting Specialist")]
        )

        # Model Training Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Model Training Specialist"),
            description="Train the XGBoost model with the optimal hyperparameters.",
            expected_output="Trained XGBoost model ready for evaluation.",
            depends_on=[agent_ids.get("Model Optimization Specialist")]
        )

        # Model Evaluation Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Model Evaluation Specialist"),
            description="Evaluate the trained model using appropriate metrics for anomaly detection.",
            expected_output="Comprehensive evaluation results including ROC-AUC, PR-AUC, and class-specific metrics.",
            depends_on=[agent_ids.get("Model Training Specialist")]
        )

        # Feature Analysis Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Feature Analysis Specialist"),
            description="Analyze feature importance from the trained model and suggest improvements.",
            expected_output="Feature importance analysis and suggestions for improvement.",
            depends_on=[agent_ids.get("Model Training Specialist")]
        )

        # Quality Assessment Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Quality Assessment Specialist"),
            description="Assess the overall quality of the model and determine if it meets production standards. " +
                        "Provide feedback for improvement if needed.",
            expected_output="Quality assessment report with go/no-go decision and feedback for improvement.",
            depends_on=[
                agent_ids.get("Model Evaluation Specialist"),
                agent_ids.get("Feature Analysis Specialist")
            ]
        )

        # Save the workflow
        self.workflows[workflow.id] = workflow

        return workflow.to_dict()

    def save_workflows_to_file(self, file_path):
        """Save all workflows to a JSON file."""
        workflows_data = [workflow.to_dict() for workflow in self.workflows.values()]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(workflows_data, f, indent=2)

    def load_workflows_from_file(self, file_path):
        """Load workflows from a JSON file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                workflows_data = json.load(f)

            # Clear existing workflows and load from file
            self.workflows = {}
            for workflow_data in workflows_data:
                workflow = Workflow.from_dict(workflow_data)
                self.workflows[workflow.id] = workflow

            return True
        return False