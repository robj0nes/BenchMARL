#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from benchmarl.eval_results import load_and_merge_json_dicts, Plotting

from matplotlib import pyplot as plt


def run_benchmark() -> List[str]:
    from benchmarl.algorithms import MaddpgConfig, MappoConfig, QmixConfig
    from benchmarl.benchmark import Benchmark
    from benchmarl.environments import VmasTask
    from benchmarl.experiment import ExperimentConfig
    from benchmarl.models.gnn import GnnConfig
    from benchmarl.models.mlp import MlpConfig

    # Configure experiment
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.save_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    experiment_config.loggers = []
    experiment_config.max_n_iters = 1500

    # Configure benchmark
    tasks = [VmasTask.PAINTING.get_from_yaml()]
    algorithm_configs = [
        MaddpgConfig.get_from_yaml(),
        MappoConfig.get_from_yaml(),
        QmixConfig.get_from_yaml(),
    ]
    model_config = GnnConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        seeds={0, 1},
        experiment_config=experiment_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
    )

    # For each experiment, run it and get its output file name
    experiments = benchmark.get_experiments()
    experiments_json_files = []
    for experiment in experiments:
        exp_json_file = str(
            Path(experiment.folder_name) / Path(experiment.name + ".json")
        )
        experiments_json_files.append(exp_json_file)
        experiment.run()
    return experiments_json_files


if __name__ == "__main__":
    # Uncomment this to rerun the benchmark that generates the files
    # experiments_json_files = run_benchmark()

    # Otherwise create a list of experiment json files.
    all_groups = dict()
    for groups in ['3A3G', '4A4G', '5A5G']:
        exp_dir = f"/Users/vd20433/FARSCOPE/FirstYearProject/BenchMARL/experiments/private/report/grouped/{groups}"
        all_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(exp_dir)) for f in fn]
        experiments_json_files = [x for x in all_files if '.json' in x and 'wandb' not in x]
        mean_len_csvs = [pd.read_csv(x) for x in all_files if '.csv' in x]
        mean_lens = []
        for csv in mean_len_csvs:
            means = csv.iloc[:, 1].to_list()
            total_mean = sum(means) / len(means)
            mean_lens.append(csv.iloc[:, 1].to_list() + [total_mean])

        mean_lens_np = np.array(mean_lens)
        av_mean_lens = np.mean(mean_lens_np, axis=0)

        final_reward = 0.6
        raw_dict = load_and_merge_json_dicts(experiments_json_files)
        steps = dict()
        min_mean = np.inf
        max_mean = -np.inf
        for i, algo in enumerate(raw_dict["vmas"]["painting"].keys()):
            num_seeds = len(raw_dict["vmas"]["painting"][algo])
            steps[algo] = []
            for j, step in enumerate(raw_dict["vmas"]["painting"][algo]["seed_0"].keys()):
                if 'absolute' not in step:
                    for return_type in raw_dict["vmas"]["painting"][algo]["seed_0"][step]:
                        if 'step_count' not in return_type:
                            # Collect all values for this timestep across seeds.
                            vals = np.array([raw_dict["vmas"]["painting"][algo][f"seed_{s}"][step][return_type] for s in
                                             range(num_seeds)])

                            # Extract the mean, min and max
                            mean_vals = np.mean(vals, axis=0)
                            min_vals = np.min(vals, axis=0)
                            max_vals = np.max(vals, axis=0)

                            # Account for early finish due to success:
                            #   Add projected reward as if episode had run for full length
                            mean_vals = [x + final_reward * (mean_lens[i][0] - mean_lens[i][j - 1]) for x in mean_vals]
                            min_vals = [x + final_reward * (mean_lens[i][0] - mean_lens[i][j - 1]) for x in min_vals]
                            max_vals = [x + final_reward * (mean_lens[i][0] - mean_lens[i][j - 1]) for x in max_vals]

                            if return_type == 'return':
                                if np.mean(mean_vals) < min_mean:
                                    min_mean = np.mean(mean_vals)
                                if np.mean(mean_vals) > max_mean:
                                    max_mean = np.mean(mean_vals)
                                steps[algo].append((np.mean(mean_vals), np.mean(min_vals), np.mean(max_vals)))
                else:
                    pass

        for key in steps.keys():
            for i, x in enumerate(steps[key]):
                steps[key][i] = (x - min_mean) / (max_mean - min_mean)

        all_groups[groups] = steps

    fig, ax = plt.subplots()
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Normalised Mean Return")
    plot_interval = 5
    for g in all_groups.keys():
        for i, key in enumerate(all_groups[g].keys()):
            data = np.array(all_groups[g][key])
            xs = 6000 * np.arange(0, len(all_groups[g][key]))
            ax.plot(
                xs[::plot_interval],
                data[:, 0][::plot_interval],
                label=f"{key.upper()}_{g}",
                marker='.',
                alpha=0.5)
            color = plt.gca().lines[-1].get_color()

            ax.fill_between(xs[::plot_interval],
                            data[:, 1][::plot_interval],
                            data[:, 2][::plot_interval],
                            alpha=0.3,
                            color=color)
    ax.legend()
    plt.show()

    # # Load and process experiment outputs
    # processed_data = Plotting.process_data(raw_dict)
    # (
    #     environment_comparison_matrix,
    #     sample_efficiency_matrix,
    # ) = Plotting.create_matrices(processed_data, env_name="vmas")

    # # # Plotting
    # # Plotting.performance_profile_figure(
    # #     environment_comparison_matrix=environment_comparison_matrix
    # # )
    # # Plotting.aggregate_scores(
    # #     environment_comparison_matrix=environment_comparison_matrix
    # # )
    # # Plotting.environemnt_sample_efficiency_curves(
    # #     sample_effeciency_matrix=sample_efficiency_matrix
    # # )
    # Plotting.task_sample_efficiency_curves(
    #     processed_data=processed_data, env="vmas", task="painting"
    # )
    # # Plotting.probability_of_improvement(
    # #     environment_comparison_matrix,
    # #     algorithms_to_compare=[["maddpg SPP=F SPC=F", "maddpg SPP=T SPC=T"]],
    # # )
    # plt.show()
