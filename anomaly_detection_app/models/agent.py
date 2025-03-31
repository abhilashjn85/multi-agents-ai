import uuid
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Agent:
    """
    Represents an AI agent in the workflow.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: str = ""
    goal: str = ""
    backstory: str = ""
    verbose: bool = True
    allow_delegation: bool = True
    max_iterations: int = 5
    communication_threshold: float = 0.7

    # Additional parameters for more advanced configurations
    llm_model: str = "mistral-7b-inst-2252b"
    temperature: float = 0.7
    max_tokens: int = 1000

    # Custom attributes for domain-specific functionality
    domain_knowledge: dict = field(default_factory=dict)

    def to_dict(self):
        """Convert the Agent to a dictionary."""
        return asdict(self)

    def to_json(self):
        """Convert the Agent to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        """Create an Agent from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str):
        """Create an Agent from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_crew_agent(self, agent_dict):
        """
        Convert to a CrewAI Agent object for use in the actual workflow.
        This method would import and return a CrewAI Agent with the
        appropriate parameters.
        """
        # In a real implementation, this would create and return a CrewAI Agent
        # For now, just return the dictionary representation
        from crewai import Agent  # Import CrewAI agent

        return Agent(name=agent_dict["name"], role=agent_dict["role"])
