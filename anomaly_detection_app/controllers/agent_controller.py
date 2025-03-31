import os
import json
import uuid
from anomaly_detection_app.models.agent import Agent


class AgentController:
    """
    Controller for managing agents.
    """

    def __init__(self):
        """Initialize the agent controller."""
        self.agents = {}
        self._load_default_agents()

    def _load_default_agents(self):
        """Load the default agents based on the original code."""
        # Define fixed UUIDs for default agents to ensure consistency99
        # This is critical for proper agent references across controllers
        default_agent_ids = {
            "Data Understanding Specialist": "00000000-0000-0000-0000-000000000001",
            "Data Preprocessing Engineer": "00000000-0000-0000-0000-000000000002",
            "Feature Engineering Specialist": "00000000-0000-0000-0000-000000000003",
            "Data Splitting Specialist": "00000000-0000-0000-0000-000000000004",
            "Model Optimization Specialist": "00000000-0000-0000-0000-000000000005",
            "Model Training Specialist": "00000000-0000-0000-0000-000000000006",
            "Model Evaluation Specialist": "00000000-0000-0000-0000-000000000007",
            "Feature Analysis Specialist": "00000000-0000-0000-0000-000000000008",
            "Quality Assessment Specialist": "00000000-0000-0000-0000-000000000009",
        }

        # Define the roles and responsibilities from the original code
        default_agents = [
            {
                "id": default_agent_ids["Data Understanding Specialist"],
                "name": "Data Understanding Specialist",
                "role": "Data Understanding Specialist",
                "goal": "Analyze data and validate configuration compatibility",
                "backstory": "You are an expert in financial dispute data analysis with specialization in "
                + "anomaly detection. You understand the nuances of financial dispute transaction "
                + "data and can quickly identify potential issues in data quality.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Data Preprocessing Engineer"],
                "name": "Data Preprocessing Engineer",
                "role": "Data Preprocessing Engineer",
                "goal": "Transform raw data into processable format",
                "backstory": "You are a skilled data engineer specialized in preparing financial dispute data "
                + "for machine learning models. You excel at handling missing values, "
                + "transforming sequences, and implementing domain-specific rules.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Feature Engineering Specialist"],
                "name": "Feature Engineering Specialist",
                "role": "Feature Engineering Specialist",
                "goal": "Create optimal features for anomaly detection",
                "backstory": "You are an expert in creating machine learning features that capture "
                + "patterns in financial dispute transaction data. You understand the importance "
                + "of sequence representations and can craft features that highlight "
                + "anomalous behavior.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Data Splitting Specialist"],
                "name": "Data Splitting Specialist",
                "role": "Data Splitting Specialist",
                "goal": "Create optimal train/test splits with balanced anomaly ratios",
                "backstory": "You are an expert in handling imbalanced datasets for anomaly detection. "
                + "You understand the importance of proper data splitting and can find the "
                + "optimal anomaly ratio for training effective models.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Model Optimization Specialist"],
                "name": "Model Optimization Specialist",
                "role": "Model Optimization Specialist",
                "goal": "Find optimal model hyperparameters",
                "backstory": "You are an expert in genetic algorithms and hyperparameter optimization. "
                + "You can efficiently navigate large parameter spaces to find the best "
                + "configuration for XGBoost models in anomaly detection.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Model Training Specialist"],
                "name": "Model Training Specialist",
                "role": "Model Training Specialist",
                "goal": "Train robust anomaly detection models",
                "backstory": "You are an expert in training machine learning models for financial "
                + "fraud detection. You understand the nuances of XGBoost and can ensure "
                + "models converge optimally without overfitting.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Model Evaluation Specialist"],
                "name": "Model Evaluation Specialist",
                "role": "Model Evaluation Specialist",
                "goal": "Evaluate model performance with appropriate metrics",
                "backstory": "You are an expert in evaluating anomaly detection models in financial "
                + "domains. You understand the importance of both precision and recall in "
                + "fraud detection and can interpret complex performance metrics.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Feature Analysis Specialist"],
                "name": "Feature Analysis Specialist",
                "role": "Feature Analysis Specialist",
                "goal": "Analyze feature importance and suggest improvements",
                "backstory": "You are an expert in interpreting machine learning models and understanding "
                + "feature contributions. You can identify the most important features for "
                + "anomaly detection and suggest improvements to feature engineering.",
                "verbose": True,
                "allow_delegation": False,
            },
            {
                "id": default_agent_ids["Quality Assessment Specialist"],
                "name": "Quality Assessment Specialist",
                "role": "Quality Assessment Specialist",
                "goal": "Ensure the final model meets quality standards",
                "backstory": "You are the final gatekeeper for model quality in financial fraud detection. "
                + "You have extensive experience in production machine learning systems and "
                + "can determine if a model is ready for deployment or needs further refinement.",
                "verbose": True,
                "allow_delegation": False,
            },
        ]

        # Create Agent objects from the default configurations
        for agent_config in default_agents:
            agent = Agent.from_dict(agent_config)
            self.agents[agent.id] = agent

    def get_available_agents(self):
        """Get all available agents."""
        return [agent.to_dict() for agent in self.agents.values()]

    def get_agent(self, agent_id):
        """Get a specific agent by ID."""
        if agent_id in self.agents:
            return self.agents[agent_id].to_dict()
        return None

    def update_agent(self, agent_id, data):
        """Update an agent's configuration."""
        if agent_id in self.agents:
            # Update the agent with the new data
            agent = self.agents[agent_id]
            for key, value in data.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            return agent.to_dict()
        return None

    def create_agent(self, data):
        """Create a new agent."""
        # Ensure the agent has a unique ID
        if "id" not in data or not data["id"]:
            data["id"] = str(uuid.uuid4())

        agent = Agent.from_dict(data)
        self.agents[agent.id] = agent
        return agent.to_dict()

    def delete_agent(self, agent_id):
        """Delete an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    def save_agents_to_file(self, file_path):
        """Save all agents to a JSON file."""
        agents_data = [agent.to_dict() for agent in self.agents.values()]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(agents_data, f, indent=2)

    def load_agents_from_file(self, file_path):
        """Load agents from a JSON file."""
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                agents_data = json.load(f)

            # Clear existing agents and load from file
            self.agents = {}
            for agent_data in agents_data:
                agent = Agent.from_dict(agent_data)
                self.agents[agent.id] = agent

            return True
        return False

    def to_crew_agent(self, agent_dict):
        """Convert agent dictionary to CrewAI Agent."""
        from crewai import Agent as CrewAgent

        # Create a CrewAI Agent from the agent dictionary
        return CrewAgent(
            role=agent_dict["role"],
            goal=agent_dict["goal"],
            backstory=agent_dict["backstory"],
            verbose=agent_dict.get("verbose", True),
            allow_delegation=agent_dict.get("allow_delegation", True),
        )

    def get_agent_map(self):
        """Get a mapping of agent names to IDs."""
        return {agent.name: agent.id for agent in self.agents.values()}
