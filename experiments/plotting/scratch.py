import ast
import os
import shutil

if __name__ == '__main__':
    exp_dir = "/Users/vd20433/FARSCOPE/FirstYearProject/Documents/report/experiments/one_hot/2024-09-12"
    run_dir = f"{exp_dir}/runs"
    group_dir = f"{exp_dir}/grouped"
    for run in os.listdir(run_dir):
        if 'DS' not in run:
            wandb_run = [x for x in os.listdir(os.path.join(run_dir, run, "wandb")) if 'run' in x][0]
            log_lines = open(os.path.join(run_dir, run, "wandb", wandb_run, "logs", "debug.log"), "r").readlines()
            for l in log_lines:
                if "config_callback()" in l:
                    src_dir = os.path.join(run_dir, run)
                    config = ast.literal_eval('{' + l.split("{", 1)[-1])
                    n_agents = config['task_config']['n_agents']
                    n_goals = config['task_config']['n_goals']
                    dest_dir = os.path.join(group_dir, f"{n_agents}A{n_goals}G")
                    shutil.copytree(src_dir, os.path.join(dest_dir, run), dirs_exist_ok=True, symlinks=False)
