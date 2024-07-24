import os
from pathlib import Path

import torch
import vmas

from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import GnnConfig, MlpConfig
from tensordict import TensorDict

from torchrl.envs.utils import ExplorationType, set_exploration_type


def get_policy():
    current_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    config_folder = current_folder / "yaml"

    config = ExperimentConfig.get_from_yaml(str(config_folder / "base_experiment.yaml"))
    config.restore_file = str(current_folder / "checkpoint_6000000.pt")

    experiment = Experiment(
        config=config,
        task=VmasTask.PAINTING.get_from_yaml(
            str(config_folder / "painting.yaml")
        ),
        algorithm_config=MaddpgConfig.get_from_yaml(str(config_folder / "maddpg.yaml")),
        model_config=GnnConfig.get_from_yaml(str(config_folder / "gnn.yaml")),
        critic_model_config=MlpConfig.get_from_yaml(
            str(config_folder / "mlp.yaml")
        ),
        seed=0,
    )

    return experiment.policy


def run_policy(policy, nav_obs, mix_obs):
    n_agents = 3

    # These are he input args
    # pos = torch.zeros((1, n_agents, 2), dtype=torch.float)
    # vel = torch.zeros((1, n_agents, 2), dtype=torch.float)
    #
    # goal = pos.clone()
    #
    # rel_goal_pos = pos - goal
    #
    # obs = torch.cat([pos, vel, rel_goal_pos], dim=-1)
    td = TensorDict(
        {"nav_agents": TensorDict({"observation": nav_obs}, batch_size=[1, n_agents]),
         "mix_agents": TensorDict({"observation": mix_obs}, batch_size=[1, n_agents])},
        batch_size=[1]
    )
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        out_td = policy(td)

    nav_actions = out_td.get(("nav_agents", "action"))
    mix_actions = out_td.get(("mix_agents", "action"))
    return nav_actions, mix_actions  # shape 1 (should squeeze), n_agents, 2


if __name__ == "__main__":
    n_agents = 3
    group_map = {
        'nav_agents': ['nav-agent_0', 'nav-agent_1', 'nav-agent_2'],
        'mix_agents': ['mix-agent_0', 'mix-agent_1', 'mix-agent_2']
    }
    policy = get_policy()

    env = vmas.make_env(
        scenario="painting",
        num_envs=1,
        continuous_actions=True,
        # Environment specific variables
        n_agents=n_agents,
        n_goals=10,
        pos_shaping=True,
        mix_shaping=True,
        learn_mix=True,
        learn_coms=True,
        coms_proximity=5,
        task_type="full",
        max_steps=500,
        group_map=group_map,
        knowledge_shape=(2, 3),
        clamp_actions=True,
        agent_collision_penalty=-0.2,
        env_collision_penalty=-0.2,
        final_pos_reward=0.2,
        final_mix_reward=0.2,
        multi_head=True
    )
    obs = env.reset()
    nav_obs = torch.stack(obs[:3], dim=-2)
    mix_obs = torch.stack(obs[3:], dim=-2)
    frame_list = []
    for _ in range(500):
        nav_actions, mix_actions = run_policy(policy, nav_obs, mix_obs)
        nav_actions = nav_actions.unbind(-2)
        mix_actions = mix_actions.unbind(-2)
        test = nav_actions + mix_actions
        obs, rews, dones, info = env.step(nav_actions + mix_actions)
        nav_obs = torch.stack(obs[:3], dim=-2)
        mix_obs = torch.stack(obs[3:], dim=-2)
        frame = env.render(
            mode="rgb_array",
            visualize_when_rgb=True,
        )
        frame_list.append(frame)

    vmas.simulator.utils.save_video(
        "painting", frame_list, fps=1 / env.scenario.world.dt
    )
