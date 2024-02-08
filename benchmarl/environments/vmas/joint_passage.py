
from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    heterogeneous: bool = MISSING
    centralised_critic: bool = MISSING
    use_mlp: bool = MISSING