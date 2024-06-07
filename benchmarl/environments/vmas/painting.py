from dataclasses import dataclass, MISSING
from typing import Literal


@dataclass
class TaskConfig:
    task_type: str = MISSING

    max_steps: int = MISSING
    n_agents: int = MISSING
    n_goals: int = MISSING
    observe_other_agents: bool = MISSING
    coms_proximity: float = MISSING

    final_pos_reward: float = MISSING
    final_mix_reward: float = MISSING
    pos_shaping: bool = MISSING
    mix_shaping: bool = MISSING

    agent_collision_penalty: float = MISSING
    env_collision_penalty: float = MISSING

