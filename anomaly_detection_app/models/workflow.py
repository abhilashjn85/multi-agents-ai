import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class Connection:
    """
    Represents a connection between two agents in the workflow.
    """

    source: str  # Source agent ID
    target: str  # Target agent ID
    label: str = ""
    type: str = "default"  # Type of connection: default, conditional, async, etc.
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        """Convert the Connection to a dictionary."""
        return asdict(self)


@dataclass
class TaskDefinition:
    """
    Represents a task definition for an agent in the workflow.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    description: str = ""
    expected_output: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(
        default_factory=list
    )  # List of task IDs that this task depends on

    def to_dict(self):
        """Convert the TaskDefinition to a dictionary."""
        return asdict(self)


@dataclass
class Workflow:
    """
    Represents a workflow of agents and their connections.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Workflow"
    description: str = ""
    created_at: str = field(default_factory=lambda: str(uuid.utcnow()))
    updated_at: str = field(default_factory=lambda: str(uuid.utcnow()))

    # Agents in this workflow
    agent_ids: List[str] = field(default_factory=list)

    # Connections between agents
    connections: List[Connection] = field(default_factory=list)

    # Task definitions
    tasks: List[TaskDefinition] = field(default_factory=list)

    # Workflow settings
    process_type: str = "sequential"  # sequential or parallel
    max_iterations: int = 10
    communication_threshold: float = 0.7

    def to_dict(self):
        """Convert the Workflow to a dictionary."""
        data = asdict(self)
        data["connections"] = [conn.to_dict() for conn in self.connections]
        data["tasks"] = [task.to_dict() for task in self.tasks]
        return data

    def to_json(self):
        """Convert the Workflow to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        """Create a Workflow from a dictionary."""
        connections = [Connection(**conn) for conn in data.pop("connections", [])]
        tasks = [TaskDefinition(**task) for task in data.pop("tasks", [])]
        workflow = cls(**data)
        workflow.connections = connections
        workflow.tasks = tasks
        return workflow

    @classmethod
    def from_json(cls, json_str):
        """Create a Workflow from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def add_agent(self, agent_id):
        """Add an agent to the workflow."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)

    def remove_agent(self, agent_id):
        """Remove an agent from the workflow."""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
            # Remove connections involving this agent
            self.connections = [
                conn
                for conn in self.connections
                if conn.source != agent_id and conn.target != agent_id
            ]
            # Remove tasks for this agent
            self.tasks = [task for task in self.tasks if task.agent_id != agent_id]

    def add_connection(
        self, source_id, target_id, label="", conn_type="default", conditions=None
    ):
        """Add a connection between two agents."""
        if conditions is None:
            conditions = {}
        connection = Connection(
            source=source_id,
            target=target_id,
            label=label,
            type=conn_type,
            conditions=conditions,
        )
        self.connections.append(connection)
        return connection

    def remove_connection(self, source_id, target_id):
        """Remove a connection between two agents."""
        self.connections = [
            conn
            for conn in self.connections
            if conn.source != source_id or conn.target != target_id
        ]

    def add_task(
        self, agent_id, description, expected_output="", context=None, depends_on=None
    ):
        """Add a task for an agent."""
        if context is None:
            context = {}
        if depends_on is None:
            depends_on = []
        task = TaskDefinition(
            agent_id=agent_id,
            description=description,
            expected_output=expected_output,
            context=context,
            depends_on=depends_on,
        )
        self.tasks.append(task)
        return task
