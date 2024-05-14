import os
from typing import List

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.gather_trajectories import load_data

plt.rcParams["font.family"] = "Times New Roman"


def parse_tensorboard(path, scalars:List[str]):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    for s in scalars:
        assert s in ea.Tags()["scalars"], f"Scalar {s} not found in event accumulator"

    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}



# create 3 plots, first is k-step mse, second is 10-step mse vs mass, third is 10-step mse vs mpc performance
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
to_skip = ["MLP", "FE", "FE_Residuals"]

# create labels
labels = {
    "MLP": "MLP",
    "FE_NeuralODE": "FE + NODE",
    "FE_NeuralODE_Residuals": "FE + NODE + Res.",
    "NeuralODE": "NODE",
    "FE": "FE",
    "Oracle": "Oracle",
    "FE_Residuals": "FE + Res.",
}
colors = {"MLP": "purple",
          "FE_NeuralODE": 'purple',
          "FE_NeuralODE_Residuals": "blue",
          "NeuralODE": "red",
          "FE": "brown",
          "Oracle": "green",
          "FE_Residuals": "orange",
          }
legend_labels = {}
# first plot its k-step mse
logdir = f"logs/drone/predictor"
mses = torch.load(os.path.join(logdir, "mses.pt"))
normalize = True

# for every alg type, plot the mse over the time horizon. Compute the quartiles for plotting
for alg_type, mse in mses.items():
    if alg_type in to_skip:
        continue
    # time_mse = torch.mean(mse, dim=(1,2))
    time_mse = mse.reshape(-1, mse.shape[-1])
    quartiles = np.percentile(time_mse.cpu().detach().numpy(), [25, 50, 75], axis=0)
    label = labels[alg_type]
    color = colors[alg_type]
    l, = axs[0].plot(quartiles[1], label=label, color=color)
    axs[0].fill_between(range(quartiles.shape[1]), quartiles[0], quartiles[2], alpha=0.2, color=color, linewidth=0.0,)
    legend_labels[label] = l
# axs[1, index].legend(loc='center left', bbox_to_anchor=(1, 0.5))
# axs[1, index].set_title("MSE over time horizon")
axs[0].set_xlabel("Lookahead Steps")
axs[0].set_ylabel("K-Step MSE")
# l = axs[0].legend(edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0), loc='upper left')
# l.get_frame().set_alpha(None)
# l.get_frame().set_facecolor((0, 0, 0, 0.0))
y_lim = 1.0
x_lim = 30
axs[0].set_ylim(0, y_lim)
axs[0].set_xlim(0, x_lim)


# now do second plot, 10-step mse vs mass
mses = torch.load(os.path.join(logdir, "drone_specific_k_step_mses.pt"))
predictor_losses = torch.load(os.path.join(logdir, "drone_specific_predictor_losses.pt"))

# second plot the mse for a k-step prediction, with mass on the X axis and MSE at the end on the Y axis
k = -1
for alg_type in mses:
    if alg_type in to_skip:
        print("Skipping")
        continue
    all_seeds = mses[alg_type]
    for seed_mses in all_seeds:
        color = colors[alg_type]
        hps, mse = seed_mses
        k = mse.shape[-1]
        mse = mse[:, :, -1]
        #
        # stds = mse.std(dim=-1)
        # mse = mse.mean(dim=-1) # mean over all trajectories. Leaves each env alone
        # assert len(hps) == mse.shape[0]
        # hp_mse = [(x, y) for x, y in zip(hps, mse)]
        # hp_mse = sorted(hp_mse, key=lambda x: x[0]["M"])
        # hp_stds = [(x, y) for x, y in zip(hps, stds)]
        # hp_stds = sorted(hp_stds, key=lambda x: x[0]["M"])
        # x = [x[0]["M"] for x in hp_mse]
        # y = [x[1].item() for x in hp_mse]
        # min = [y - s for y, s in zip(y, [x[1].item() for x in hp_stds])]
        # max = [y + s for y, s in zip(y, [x[1].item() for x in hp_stds])]
        # axs[1].plot(x, y, label=labels[alg_type], color=color)
        # axs[1].fill_between(x, min, max, alpha=0.3, color=color)
        quants = np.percentile(mse.cpu().detach().numpy(), [25, 50, 75], axis=1).transpose()
        hp_quants = [(x, y) for x, y in zip(hps, quants)]
        hp_quants = sorted(hp_quants, key=lambda x: x[0]["M"])
        hps = [x[0]["M"] for x in hp_quants]
        quants = [x[1] for x in hp_quants]
        axs[1].plot(hps, [q[1] for q in quants], label=labels[alg_type], color=color)
        axs[1].fill_between(hps, [q[0] for q in quants], [q[2] for q in quants], alpha=0.3, color=color, linewidth=0.0,)
        min_mass, max_mass = min(hps), max(hps)

axs[1].set_xlabel("Mass")
axs[1].set_ylabel(f"{k}-step MSE")
axs[1].set_xlim(min_mass, max_mass)
# axs[1].set_xticks([0.02, 0.024, 0.028, 0.032])
# l = axs[1].legend(edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0), loc='upper right')
# l.get_frame().set_alpha(None)
# l.get_frame().set_facecolor((0, 0, 0, 0.0))



# now do third plot, mpc slew rates
ax = axs[2]
def get_slews(actions):
    action_change = actions[:, :, 1:] - actions[:, :, :-1]
    action_change = action_change ** 2
    action_change = action_change.mean(dim=3)
    return action_change

env_str = 'drone'
load_dir = "logs/drone/predictor"
policy_type = 'random'

_, _, _, _, _, _, _, test_hidden_params = load_data(env_str, policy_type, normalize=False,
                                                    get_groundtruth_hidden_params=True)

# fetch all the subdirectories
alg_dirs = [os.path.join(load_dir, x) for x in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, x))]

results = {}
for alg_dir in alg_dirs:
    # load the subdir. There should be only 1
    subdirs = [x for x in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, x))]
    assert len(subdirs) == 1, f"Expected 1 subdir, got {len(subdirs)}"
    alg_type = subdirs[0]
    alg_dir = os.path.join(alg_dir, alg_type, policy_type)

    # try to load the results
    try:
        states = torch.load(os.path.join(alg_dir, 'mpc_states.pt'))
        actions = torch.load(os.path.join(alg_dir, 'mpc_actions.pt'))
    except:
        print(f"Could not load results for {alg_dir}")
        continue

    # skip
    if states.shape != (40, 5, 101, 13):  # these are tempory test runs, skip them
        continue

    # get results
    returns = get_slews(actions)
    results[alg_type] = returns

# now plot
for alg_type, returns in results.items():
    # sum over all timesteps
    returns = returns.sum(dim=2)
    print("Alg type", alg_type, "Mean return", returns.mean().item(), "Min return", returns.min().item(), "Max return",
          returns.max().item())

    # compute quantiles to account for crashes
    quarts = torch.quantile(returns, torch.tensor([0.0, 0.5, 1.0]), dim=1).transpose(0, 1)
    means = returns.mean(dim=1)
    stds = returns.std(dim=1)
    quarts_with_hps = [(x, test_hidden_params[i]) for i, x in enumerate(quarts)]
    means_with_hps = [(x, test_hidden_params[i]) for i, x in enumerate(means)]
    stds_with_hps = [(x, test_hidden_params[i]) for i, x in enumerate(stds)]

    # sort based on hps for graph
    quarts_with_hps.sort(key=lambda x: x[1]['M'])
    means_with_hps.sort(key=lambda x: x[1]['M'])
    stds_with_hps.sort(key=lambda x: x[1]['M'])

    # plot min, max, median
    xs = [x[1]['M'] for x in quarts_with_hps]  # [::4]
    mins = [x[0][0] for x in quarts_with_hps]  # [::4]
    medians = [x[0][1] for x in quarts_with_hps]  # [::4]
    maxs = [x[0][2] for x in quarts_with_hps]  # [::4]
    means = [x[0] for x in means_with_hps]  # [::4]
    stds = [x[0] for x in stds_with_hps]  # [::4]

    # plot
    label = labels[alg_type]
    color = colors[alg_type]
    ax.plot(xs, medians, label=label, color=color)
    ax.fill_between(xs, mins, maxs, alpha=0.2, color=color, linewidth=0.0,)

    # ax.plot(xs, means, label=alg_type)
    # ax.fill_between(xs, [m-s for m,s in zip(means, stds)], [m+s for m,s in zip(means, stds)], alpha=0.2)

# ax.set_ylim(-100, 0)
# ax.legend()
ax.set_xlabel('Mass')
ax.set_ylabel('Average Slew Rate')
print(min_mass, max_mass)
ax.set_xlim(min_mass, max_mass)
# ax.set_xticks([0.02, 0.024, 0.028, 0.032])
ax.set_ylim(0, 0.25)

# save the figure
fig.legend(loc=10,
           labels=legend_labels.keys(), handles=legend_labels.values(),
           ncol=5,
           bbox_to_anchor=(0.5, 0.02),
           edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0)
           )

plt.tight_layout(rect=(-0.01, 0.02, 1, 1.02))
plt.savefig("drone_results.pdf", dpi=300)
