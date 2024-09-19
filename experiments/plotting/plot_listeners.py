import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    res_dir = "/experiments/private/listeners-test"
    files = os.listdir(res_dir)
    mean_lens = pd.read_csv(os.path.join(res_dir, "3A_3G_RdmAllDims_MeanLens.csv")).iloc[:, 1]
    results = pd.read_csv(os.path.join(res_dir, "3A_3G_RdmAllDims_ListenersEval.csv")).iloc[:, 1]

    final_rew = 0.2
    min = np.inf
    max = 0
    adjusted_res = []
    for i in range(len(results)):
        adjusted = results[i] + (final_rew * (mean_lens[0] - mean_lens[i]))
        if adjusted < min:
            min = adjusted
        if adjusted > max:
            max = adjusted
        adjusted_res.append(adjusted)

    normalised = [((x-min)/(max-min)) for x in adjusted_res]
    steps = 6000 * np.arange(0, len(normalised))
    fig, ax = plt.subplots()
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Normalised Mean Return")
    ax.set_xticks(np.arange(0, 6000*len(normalised), 200000))
    plt.plot(steps, normalised, 'b-', label='Listeners')
    plt.show()