from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_collection_points: int = MISSING
    n_blueprints: int = MISSING
