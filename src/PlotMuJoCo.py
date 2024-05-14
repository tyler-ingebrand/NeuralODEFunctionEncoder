import os
from typing import List

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import torch

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

envs = ["HalfCheetah-v3", "Ant-v3"]


# plot both side by side
fig, axs = plt.subplots(2, 3,
                        figsize=(10, 5),
                        gridspec_kw={'width_ratios': [2.5, 7.5/2, 7.5/2], 'height_ratios':[2.5, 2.5]})

to_skip = ["MLP", "FE"]

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
# Plot images of each environment in the first column
# make the images smaller than the plots, and label them with the environment names
axs[0, 0].imshow(plt.imread("images/halfcheetah.png"))
axs[0, 0].axis("off")
axs[0, 0].set_title("Half Cheetah", y=-0.15)

axs[1, 0].imshow(plt.imread("images/ant.png"))
axs[1, 0].axis("off")
axs[1, 0].set_title("Ant", y=-0.15)

legend_labels = {}

# First do 1 step loss
for index, env in enumerate(envs):
    logdir = f"logs/{'ant' if env == 'Ant-v3' else 'cheetah'}/predictor"

    # search all subdirs
    subdirs = [x for x in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, x))]
    assert len(subdirs) > 0, "No subdirs found"


    # for each subdir, parse the tensorboard scalars
    losses = {}
    step = None
    for subdir in subdirs:
        if subdir == "old":
            continue
        # need to go two dirs down
        alg_dir = os.listdir(os.path.join(logdir, subdir))[0]
        if alg_dir in to_skip: # skipping these to declutter graph.
            continue
        data_dir = os.listdir(os.path.join(logdir, subdir, alg_dir))[0]
        print(f"Parsing: {os.path.join(logdir, subdir, alg_dir, data_dir)}")
        if alg_dir not in losses:
            losses[alg_dir] = []

        # parse
        vals = parse_tensorboard(os.path.join(logdir, subdir, alg_dir, data_dir), ["test/loss"])
        step = vals["test/loss"]["step"].to_numpy()
        loss = vals["test/loss"]["value"].to_numpy()
        losses[alg_dir].append(loss)

    # plot
    for alg, loss in losses.items():
        min_size = min([len(x) for x in loss])
        loss = [x[:min_size] for x in loss]
        loss = np.array(loss)
        # mean = np.mean(loss, axis=0)
        # std = np.std(loss, axis=0)
        quants = np.percentile(loss, [25, 50, 75], axis=0)
        median = quants[1]
        lower = quants[0]
        upper = quants[2]
        label = labels[alg]
        color = colors[alg]
        # axs[0, index].plot(step[:len(mean)], mean, label=label, color=color)
        # axs[0, index].fill_between(step[:len(mean)], mean - std, mean + std, alpha=0.2, color=color)
        line,  = axs[index, 1].plot(step[:len(median)], median, label=label, color=color)
        axs[index, 1].fill_between(step[:len(median)], lower, upper, alpha=0.2, color=color, linewidth=0.0,)
        legend_labels[label] = line

    axs[index, 1].set_xlim([0, 1000])
    axs[index, 1].set_ylim([0, 0.25])
    axs[index, 1].set_ylabel("1-Step MSE")


    # give more space, x and y axis labels are blocked off
    # axs[index, 1].set_title(f"{env.replace('-v3', '')}")
    if index == 1:
        axs[index, 1].set_xlabel("Gradient Updates")



# next do k-step loss after training
for index, env in enumerate(envs):

    logdir = f"logs/{'ant' if env == 'Ant-v3' else 'cheetah'}/predictor"
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
        axs[index, 2].plot(quartiles[1], label=label, color=color)
        axs[index, 2].fill_between(range(quartiles.shape[1]), quartiles[0], quartiles[2], alpha=0.2, color=color, linewidth=0.0,)

    # axs[index, 2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[index, 2].set_title("MSE over time horizon")
    if index == 1:
        axs[index, 2].set_xlabel("Lookahead Steps")
    axs[index, 2].set_ylabel("K-Step MSE")
    # if index == 1: # ste background to be transparent

    # if index == 1:
        # l = axs[index, 2].legend(loc='center left', bbox_to_anchor=(1, 0.75), edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0), ncol=1)
        # l = axs[index, 2].legend(loc='center left', bbox_to_anchor=(-1, -1.75), edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0), ncol=5)
        # l.get_frame().set_alpha(None)
        # l.get_frame().set_facecolor((0, 0, 0, 0.0))
    y_lim, x_lim = None, None
    if normalize:
        if env == "HalfCheetah-v3":
            y_lim = 1.0
            x_lim = 99
        elif env == "drone":
            y_lim = 1.0
            x_lim = 30
        else:
            y_lim = 1.0
            x_lim = 99
    if not normalize:
        y_lim = 15.0
        x_lim = 99
    axs[index, 2].set_ylim(0, y_lim)
    axs[index, 2].set_xlim(0, x_lim)

# reorder labels and handles to be in
# NODE > FE + RES > FE + NODE > FE + NODE + RES > oracle
order = ["NODE", "FE + Res.", "FE + NODE", "FE + NODE + Res.", "Oracle"]
legend_labels = {k: legend_labels[k] for k in order}


fig.legend(loc=10,
           labels=legend_labels.keys(), handles=legend_labels.values(),
           ncol=5,
           bbox_to_anchor=(0.5, 0.02),
           edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0)
           )

plt.tight_layout(rect=(-0.01, 0.02, 1, 1.02))
plt.savefig("mujoco_results.pdf", dpi=300)