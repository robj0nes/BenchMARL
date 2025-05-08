from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    n_agents: int = MISSING
    max_steps: int = MISSING
