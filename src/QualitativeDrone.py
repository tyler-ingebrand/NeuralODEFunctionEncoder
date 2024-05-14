import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.PlotDrone import labels, colors
from src.gather_trajectories import load_data

plt.rcParams["font.family"] = "Times New Roman"

# going to plot three images. First is a graphic showing drone course correction. Second and third are
# trajectories loaded from data.
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# collection of data
env_str = 'drone'
load_dir = "logs/drone/predictor"
policy_type = 'random'
low_data = False

alg_dirs = [os.path.join(load_dir, x) for x in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, x))]
_, _, _, _, _, _, _, test_hidden_params = load_data(env_str, policy_type, normalize=False,
                                                    get_groundtruth_hidden_params=True)

# plot first one
ax = axs[0]
# load "./images/drone_diagram.png"
ax.imshow(plt.imread("./images/drone_diagram.png"))
ax.axis('off')


# plot second one
ax = axs[1]
# now look at an individual trajectory
# get the smallest mass index
smallest_mass_index = np.argmin([x['M'] for x in test_hidden_params])
env_index = smallest_mass_index
traj_index = 1

results = {}
for alg_dir in alg_dirs:
    # load the subdir. There should be only 1
    subdirs = [x for x in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, x))]
    assert len(subdirs) == 1, f"Expected 1 subdir, got {len(subdirs)}"
    alg_type = subdirs[0]
    alg_dir = os.path.join(alg_dir, alg_type, policy_type)

    # try to load the results
    try:
        extension = '_low_data' if alg_type != 'NeuralODE' and low_data else ''
        states = torch.load(os.path.join(alg_dir, f'mpc_states{extension}.pt'))
    except:
        print(f"Could not load results for {alg_dir}")
        continue
    states = states[env_index, traj_index]
    results[alg_type] = states
mass = test_hidden_params[env_index]['M']


# now plot z axis over time
xs = np.arange(0, 101,dtype=np.float32)
xs *= 0.05
for alg_type, states in results.items():
    label = labels[alg_type]
    color = colors[alg_type]
    ax.plot(xs, states[:, 4], label=label, color=color)

# plot dotted line at z=1 for goal location
ax.plot([0, max(xs)], [1, 1], 'k--', label='Goal')
ax.set_xlabel('Time (Seconds)')
ax.set_ylabel('Z (Meters)')
ax.set_title(f"Low Mass Trajectory", loc='right', y=0.0, x=0.97)
ax.set_xlim(0, 5)





# plot third one
ax = axs[2]
# now look at an individual trajectory
# get the smallest mass index
largest_mass_index = np.argmax([x['M'] for x in test_hidden_params])
env_index = largest_mass_index
traj_index = 1

results = {}
for alg_dir in alg_dirs:
    # load the subdir. There should be only 1
    subdirs = [x for x in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, x))]
    assert len(subdirs) == 1, f"Expected 1 subdir, got {len(subdirs)}"
    alg_type = subdirs[0]
    alg_dir = os.path.join(alg_dir, alg_type, policy_type)

    # try to load the results
    try:
        extension = '_low_data' if alg_type != 'NeuralODE' and low_data else ''
        states = torch.load(os.path.join(alg_dir, f'mpc_states{extension}.pt'))
    except:
        print(f"Could not load results for {alg_dir}")
        continue
    states = states[env_index, traj_index]
    results[alg_type] = states
mass = test_hidden_params[env_index]['M']


# now plot z axis over time
legend_labels = {}
for alg_type, states in results.items():
    label = labels[alg_type]
    color = colors[alg_type]
    l, = ax.plot(xs, states[:, 4], label=label, color=color)
    legend_labels[label] = l

# plot dotted line at z=1 for goal location
l, = ax.plot([0, max(xs)], [1, 1], 'k--', label='Goal')
legend_labels['Goal'] = l
ax.set_xlabel('Time (Seconds)')
ax.set_ylabel('Z (Meters)')
ax.set_title(f"High Mass Trajectory", loc='right', y=0.0, x=0.97)
# ax.legend(loc="lower right")
ax.set_xlim(0, 5)


fig.legend(loc=10,
           labels=legend_labels.keys(), handles=legend_labels.values(),
           ncol=5,
           bbox_to_anchor=(0.5, 0.02),
           edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0)
           )

plt.tight_layout(rect=(-0.01, 0.02, 1, 1.02))
plt.savefig("qualitative_drone_results.pdf", dpi=300)
