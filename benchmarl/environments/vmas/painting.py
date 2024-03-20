from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_goals: int = MISSING
    final_reward: float = MISSING
    pos_shaping: bool = MISSING
