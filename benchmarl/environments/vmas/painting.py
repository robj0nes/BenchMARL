from dataclasses import dataclass, MISSING
from typing import Dict


@dataclass
class TaskConfig:
    task_type: str = MISSING

    max_steps: int = MISSING
    n_agents: int = MISSING
    n_goals: int = MISSING
    multi_head: bool = MISSING

    group_map: Dict = MISSING

    observe_other_agents: bool = MISSING
    random_knowledge: bool = MISSING
    learn_mix: bool = MISSING
    learn_coms: bool = MISSING
    per_agent_final: bool = MISSING

    coms_proximity: float = MISSING
    mixing_thresh: float = MISSING

    final_pos_reward: float = MISSING
    final_mix_reward: float = MISSING
    goal_completion_reward: float = MISSING

    pos_shaping: bool = MISSING
    pos_shaping_factor: float = MISSING
    mix_shaping: bool = MISSING
    mix_shaping_factor: float = MISSING

    agent_collision_penalty: float = MISSING
    env_collision_penalty: float = MISSING

