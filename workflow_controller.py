import os
import json
import uuid
import datetime
from anomaly_detection_app.models.workflow import Workflow, Connection, TaskDefinition


class WorkflowController:
    """
    Controller for managing workflows.
    """

    def __init__(self, agent_controller=None):
        """Initialize the workflow controller."""
        self.workflows = {}
        self.agent_controller = agent_controller

    def register_agent_controller(self, agent_controller):
        """Register the agent controller."""
        self.agent_controller = agent_controller
        # Initialize default workflows after setting the agent controller
        self.initialize_default_workflows()

    def initialize_default_workflows(self):
        """Initialize default workflows after agent controller is registered."""
        if not self.agent_controller:
            print(
                "WARNING: No agent controller registered, default workflow may have missing agents"
            )

        # Clear existing workflows to avoid duplicates
        self.workflows = {}
        self._load_default_workflows()

    def _load_default_workflows(self):
        """Load the default workflow based on the original code."""
        # Create a default workflow for anomaly detection
        workflow_id = str(uuid.uuid4())
        default_workflow = Workflow(
            id=workflow_id,
            name="Anomaly Detection Workflow",
            description="Default workflow for anomaly detection using XGBoost and LLM agents",
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
            process_type="sequential",
        )

        # If agent controller is available, add default agents
        if self.agent_controller:
            agent_map = self.agent_controller.get_agent_map()
            if agent_map:
                # Add agent IDs to the workflow
                for agent_name, agent_id in agent_map.items():
                    default_workflow.add_agent(agent_id)

                # Create default connections between agents
                self._add_default_connections(default_workflow, agent_map)

                # Create default tasks
                self._add_default_tasks(default_workflow, agent_map)

        self.workflows[default_workflow.id] = default_workflow
        print(f"Default workflow created with ID: {default_workflow.id}")
        print(
            f"Workflow has {len(default_workflow.agent_ids)} agents and {len(default_workflow.tasks)} tasks"
        )

    def _add_default_connections(self, workflow, agent_map):
        """Add default connections between agents."""
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
            "Quality Assessment Specialist",
        ]

        # Add connections between agents
        for i in range(len(agent_sequence) - 1):
            source_name = agent_sequence[i]
            target_name = agent_sequence[i + 1]

            if source_name in agent_map and target_name in agent_map:
                source_id = agent_map[source_name]
                target_id = agent_map[target_name]
                workflow.add_connection(source_id, target_id)

    def _add_default_tasks(self, workflow, agent_map):
        """Add default tasks for agents."""
        # Data Understanding Specialist
        if "Data Understanding Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Data Understanding Specialist"],
                description="Analyze the data and validate configuration compatibility. "
                + "Identify any potential issues for anomaly detection.",
                expected_output="A detailed analysis of the data structure and quality issues.",
            )

        # Data Preprocessing Engineer
        if "Data Preprocessing Engineer" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Data Preprocessing Engineer"],
                description="Process the raw data according to the configuration including sequence processing, "
                + "categorical encoding, and implementing anomaly rules.",
                expected_output="Processed data ready for feature engineering.",
                depends_on=[agent_map.get("Data Understanding Specialist", "")],
            )

        # Feature Engineering Specialist
        if "Feature Engineering Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Feature Engineering Specialist"],
                description="Create optimal features for anomaly detection from the processed data including "
                + "TF-IDF embeddings, categorical features, and anomaly indicators.",
                expected_output="Feature matrix ready for model training.",
                depends_on=[agent_map.get("Data Preprocessing Engineer", "")],
            )

        # Data Splitting Specialist
        if "Data Splitting Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Data Splitting Specialist"],
                description="Find the optimal anomaly ratio for training and create the best train/test split.",
                expected_output="Optimal train/test split with appropriate anomaly ratio.",
                depends_on=[agent_map.get("Feature Engineering Specialist", "")],
            )

        # Model Optimization Specialist
        if "Model Optimization Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Model Optimization Specialist"],
                description="Use genetic algorithm to find optimal hyperparameters for the XGBoost model.",
                expected_output="Optimal hyperparameters for XGBoost.",
                depends_on=[agent_map.get("Data Splitting Specialist", "")],
            )

        # Model Training Specialist
        if "Model Training Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Model Training Specialist"],
                description="Train the XGBoost model with the optimal hyperparameters.",
                expected_output="Trained XGBoost model ready for evaluation.",
                depends_on=[agent_map.get("Model Optimization Specialist", "")],
            )

        # Model Evaluation Specialist
        if "Model Evaluation Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Model Evaluation Specialist"],
                description="Evaluate the trained model using appropriate metrics for anomaly detection.",
                expected_output="Comprehensive evaluation results including ROC-AUC, PR-AUC, and class-specific metrics.",
                depends_on=[agent_map.get("Model Training Specialist", "")],
            )

        # Feature Analysis Specialist
        if "Feature Analysis Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Feature Analysis Specialist"],
                description="Analyze feature importance from the trained model and suggest improvements.",
                expected_output="Feature importance analysis and suggestions for improvement.",
                depends_on=[agent_map.get("Model Training Specialist", "")],
            )

        # Quality Assessment Specialist
        if "Quality Assessment Specialist" in agent_map:
            workflow.add_task(
                agent_id=agent_map["Quality Assessment Specialist"],
                description="Assess the overall quality of the model and determine if it meets production standards. "
                + "Provide feedback for improvement if needed.",
                expected_output="Quality assessment report with go/no-go decision and feedback for improvement.",
                depends_on=[
                    agent_map.get("Model Evaluation Specialist", ""),
                    agent_map.get("Feature Analysis Specialist", ""),
                ],
            )

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
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid.uuid4())

        # Set created and updated timestamps
        data["created_at"] = datetime.datetime.now().isoformat()
        data["updated_at"] = datetime.datetime.now().isoformat()

        workflow = Workflow.from_dict(data)
        self.workflows[workflow.id] = workflow
        return workflow.to_dict()

    def update_workflow(self, workflow_id, data):
        """Update a workflow."""
        if workflow_id in self.workflows:
            # Update the workflow with the new data
            workflow = self.workflows[workflow_id]

            # Handle special collections like connections and tasks
            if "connections" in data:
                connections = [Connection(**conn) for conn in data.pop("connections")]
                workflow.connections = connections

            if "tasks" in data:
                tasks = [TaskDefinition(**task) for task in data.pop("tasks")]
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
            process_type="sequential",
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
            "Quality Assessment Specialist",
        ]

        # Add connections between agents
        for i in range(len(agent_sequence) - 1):
            source_name = agent_sequence[i]
            target_name = agent_sequence[i + 1]

            if source_name in agent_ids and target_name in agent_ids:
                source_id = agent_ids[source_name]
                target_id = agent_ids[target_name]
                workflow.add_connection(source_id, target_id)

        # Add tasks for each agent (similar to the _add_default_tasks method)
        # Data Understanding Specialist
        workflow.add_task(
            agent_id=agent_ids.get("Data Understanding Specialist"),
            description="Analyze the data and validate configuration compatibility. "
            + "Identify any potential issues for anomaly detection.",
            expected_output="A detailed analysis of the data structure and quality issues.",
        )

        # Data Preprocessing Engineer
        workflow.add_task(
            agent_id=agent_ids.get("Data Preprocessing Engineer"),
            description="Process the raw data according to the configuration including sequence processing, "
            + "categorical encoding, and implementing anomaly rules.",
            expected_output="Processed data ready for feature engineering.",
            depends_on=[agent_ids.get("Data Understanding Specialist")],
        )

        # Add the rest of the tasks (similar to the previous task definitions)
        # ...

        # Save the workflow
        self.workflows[workflow.id] = workflow

        return workflow.to_dict()

    def save_workflows_to_file(self, file_path):
        """Save all workflows to a JSON file."""
        workflows_data = [workflow.to_dict() for workflow in self.workflows.values()]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(workflows_data, f, indent=2)

    def load_workflows_from_file(self, file_path):
        """Load workflows from a JSON file."""
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                workflows_data = json.load(f)

            # Clear existing workflows and load from file
            self.workflows = {}
            for workflow_data in workflows_data:
                workflow = Workflow.from_dict(workflow_data)
                self.workflows[workflow.id] = workflow

            return True
        return False
