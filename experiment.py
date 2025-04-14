import uuid
import json
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class ExperimentConfig:
    """
    Represents the configuration for an experiment.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Configuration"
    description: str = ""

    # Configuration parameters from the original code
    input_columns: Dict[str, str] = field(default_factory=dict)
    categorical_columns: List[str] = field(default_factory=list)
    multi_value_columns: List[str] = field(default_factory=list)
    output_column: str = "IS_ANOMALY"
    n_gram_range: List[int] = field(default_factory=lambda: [1, 3])
    anomaly_rules: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    model_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ga_params: Dict[str, Any] = field(default_factory=dict)
    objective: str = "binary:logistic"
    eval_metric: str = "auc"

    # LLM API settings
    llm_api_url: str = ""
    llm_model_name: str = "mistral-7b-inst-2252b"

    def to_dict(self):
        """Convert the ExperimentConfig to a dictionary."""
        return asdict(self)

    def to_json(self):
        """Convert the ExperimentConfig to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        """Create an ExperimentConfig from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str):
        """Create an ExperimentConfig from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def default_config(cls):
        """Create a default ExperimentConfig."""
        return cls(
            name="Default Configuration",
            description="Default configuration for anomaly detection",
            input_columns={
                "reason_code_sequence": "REASON_CODE_SEQUENCE",
                "liability_sequence": "LIABILITY_SEQUENCE",
                "mm_status_sequence": "MM_STATUS_SEQUENCE",
            },
            categorical_columns=[
                "REASON_CODE",
                "CLAIM_TYPE",
                "CLAIM_SUBTYPE",
                "IS_SUBSEQUENT",
            ],
            multi_value_columns=["E_TXN_INDICATOR"],
            output_column="IS_ANOMALY",
            n_gram_range=[1, 3],
            anomaly_rules={
                "MM_STATUS_SEQUENCE": [
                    {
                        "type": "repeated_step",
                        "step": "RECOVER_DISPUTED_FUNDS,SUCCESS",
                        "min_repetitions": 2,
                    }
                ]
            },
            model_params={
                "max_depth": {"min": 3, "max": 10},
                "eta": {"min": 0.01, "max": 0.3},
                "learning_rate": {"min": 0.01, "max": 0.05},
                "subsample": {"min": 0.5, "max": 1.0},
                "colsample_bytree": {"min": 0.5, "max": 1.0},
                "min_child_weight": {"min": 1, "max": 10},
                "gamma": {"min": 0, "max": 5},
                "lambda": {"min": 0.01, "max": 10},
                "alpha": {"min": 0.01, "max": 10},
            },
            ga_params={
                "population_size": 50,
                "generations": 1,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
            },
            objective="binary:logistic",
            eval_metric="auc",
        )


@dataclass
class Experiment:
    """
    Represents an experiment run.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Experiment"
    description: str = ""
    workflow_id: str = ""
    config_id: str = ""
    data_path: str = ""
    output_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    started_at: str = ""
    completed_at: str = ""
    status: str = "created"  # created, running, completed, failed

    # Runtime information
    progress: float = 0.0  # 0 to 1
    current_phase: str = ""
    current_agent: str = ""
    log_entries: List[Dict[str, Any]] = field(default_factory=list)

    # Results
    results: Dict[str, Any] = field(default_factory=dict)

    # Storage for in-memory data during execution (not serialized)
    raw_data: Any = None
    processed_data: Any = None
    label_encoders: Any = None
    features: Any = None
    feature_names: List[str] = field(default_factory=list)
    labels: Any = None
    vectorizer: Any = None
    mlb: Any = None
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    best_params: Any = None
    model: Any = None
    evaluation_results: Any = None
    feature_importance: Any = None
    quality_assessment: Any = None
    agent_mapping: Dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        """Convert the Experiment to a dictionary."""
        data = asdict(self)
        # Remove non-serializable data
        non_serializable = ['raw_data', 'processed_data', 'label_encoders', 'features', 'labels',
                            'vectorizer', 'mlb', 'X_train', 'X_test', 'y_train', 'y_test',
                            'best_params', 'model', 'evaluation_results', 'feature_importance',
                            'quality_assessment']
        for field in non_serializable:
            if field in data:
                data[field] = None
        return data

    def to_json(self):
        """Convert the Experiment to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        """Create an Experiment from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str):
        """Create an Experiment from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def add_log_entry(self, message, level="INFO", agent="", phase=""):
        """Add a log entry to the experiment."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "message": message,
            "level": level,
            "agent": agent or self.current_agent,
            "phase": phase or self.current_phase,
        }
        self.log_entries.append(entry)

    def update_status(self, status, progress=None, phase=None, agent=None):
        """Update the status of the experiment."""
        self.status = status

        if progress is not None:
            self.progress = progress

        if phase is not None:
            self.current_phase = phase

        if agent is not None:
            self.current_agent = agent

        if status == "running" and not self.started_at:
            self.started_at = datetime.datetime.now().isoformat()

        if status == "completed" or status == "failed":
            self.completed_at = datetime.datetime.now().isoformat()