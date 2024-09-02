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
    ckpt_folder = current_folder / "checkpoints"

    config = ExperimentConfig.get_from_yaml(str(config_folder / "base_experiment.yaml"))

    config.restore_file = str(ckpt_folder / "3H_3A_3G_1Hot_rdmSourceDim/checkpoint_12000000.pt")

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


def run_policy(policy, obs):
    n_agents = 3

    td = TensorDict(
        {"nav_agents": TensorDict({"observation": torch.stack(obs[:3], dim=-2)}, batch_size=[1, n_agents]),
         "speakers": TensorDict({"observation": torch.stack(obs[3:6], dim=-2)}, batch_size=[1, n_agents]),
         "listeners": TensorDict({"observation": torch.stack(obs[6:], dim=-2)}, batch_size=[1, n_agents])},
        batch_size=[1]
    )

    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        out_td = policy(td)

    actions = ()
    for key in td.keys():
        actions += out_td.get((key, "action")).unbind(-2)

    return actions # shape 1 (should squeeze), n_agents, 2


if __name__ == "__main__":
    n_agents = 3
    group_map = {
        'nav_agents': ['nav-agent_0', 'nav-agent_1', 'nav-agent_2'],
        'speakers': ['speak-agent_0', 'speak-agent_1', 'speak-agent_2'],
        'listeners': ['listen-agent_0', 'listen-agent_1', 'listen-agent_2']
    }
    policy = get_policy()

    env = vmas.make_env(
        scenario="painting",
        goals_from_image="/Users/vd20433/FARSCOPE/FirstYearProject/BenchMARL/run_models/images/8x8face.png",
        debug=False,
        num_envs=1,
        continuous_actions=True,
        # Environment specific variables
        n_agents=n_agents,
        n_goals=5,
        pos_shaping=True,
        mix_shaping=True,
        learn_mix=True,
        learn_coms=True,
        coms_proximity=5,
        coms_thresh=0.15,
        mixing_thresh=0.15,
        task_type="full",
        max_steps=1000,
        group_map=group_map,
        knowledge_shape=(2, 3),
        clamp_actions=True,
        agent_collision_penalty=-0.2,
        env_collision_penalty=-0.2,
        final_pos_reward=0.2,
        final_mix_reward=0.2,
        final_coms_reward=0.2,
        multi_head=True,
        random_knowledge=False,
        random_all_dims=False,
        random_source_dim=True,
        seed=6,
    )
    obs = env.reset()
    frame_list = []
    for _ in range(1000):
        actions = run_policy(policy, obs)
        obs, rews, dones, info = env.step(actions)
        frame = env.render(
            mode="rgb_array",
            visualize_when_rgb=True,
        )
        frame_list.append(frame)

    vmas.simulator.utils.save_video(
        "painting", frame_list, fps=1 / env.scenario.world.dt
    )
