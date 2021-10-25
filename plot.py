# Draw learning graph of single hyperparameter
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name",
    help="Butterfly Environment to use from PettingZoo",
    type=str,
    default="pistonball_v4",
    choices=[
        "pistonball_v4",
        "cooperative_pong_v3",
        "knights_archers_zombies_v7",
        "prospector_v4",
    ],
)
parser.add_argument("--n-runs", type=int, default=5)
args = parser.parse_args()

log_dir = "./data/" + args.env_name + "/"

result_per_timestep = {}

# Load data
for i in range(args.n_runs):
    run_log_dir = log_dir + "run_" + str(i) + "/"
    run_log = run_log_dir + "evaluations.npz"

    data = np.load(run_log)
    data_timesteps = data["timesteps"]
    data_results = data["results"]

    if len(result_per_timestep.keys()) == 0:
        for t in range(len(data_timesteps)):
            data_timestep = data_timesteps[t]
            data_result = data_results[t]

            # Store mean of 10 evaluations for each run.
            result_per_timestep[data_timestep] = np.mean(data_result)
    else:
        for t in range(len(data_timesteps)):
            data_timestep = data_timesteps[t]
            data_result = data_results[t]

            if data_timestep not in result_per_timestep.keys():
                print("Inconsistent time step error")
                exit()

            result_per_timestep[data_timestep] = np.append(
                result_per_timestep[data_timestep], np.mean(data_result)
            )

# Draw graph
matplotlib.use("pgf")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
    }
)

fig, ax = plt.subplots()
clrs = sns.color_palette()
with sns.axes_style("darkgrid"):
    timesteps = list(result_per_timestep.keys())

    nrow = len(timesteps)
    ncol = args.n_runs
    results = np.zeros((nrow, ncol))
    for i in range(nrow):
        results[i][:] = result_per_timestep[timesteps[i]]

    mean_results = np.mean(results, axis=1)
    std_results = np.std(results, axis=1)

    mean_spline = make_interp_spline(timesteps, mean_results)
    std_spline = make_interp_spline(timesteps, std_results)

    n_timesteps = np.linspace(0, np.max(timesteps), 500)
    n_mean_results = mean_spline(n_timesteps)
    n_std_results = std_spline(n_timesteps)

    ax.plot(
        n_timesteps,
        n_mean_results,
        c=clrs[0],
    )
    ax.fill_between(
        n_timesteps,
        n_mean_results - n_std_results,
        n_mean_results + n_std_results,
        alpha=0.3,
        facecolor=clrs[0],
    )
    ax.set_xlabel("Steps", labelpad=1)
    ax.set_ylabel("Average Total Reward", labelpad=1)
    ax.set_title(args.env_name)
    ax.margins(x=0)
    plt.tight_layout(pad=1.00)

    # plt.show()
    plt.savefig(
        "./figures/PPO_" + args.env_name + ".pgf",
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.savefig(
        "./figures/PPO_" + args.env_name + ".png",
        bbox_inches="tight",
        pad_inches=0.025,
        dpi=600,
    )
