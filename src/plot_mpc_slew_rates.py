import os

import torch
from matplotlib import pyplot as plt

from src.gather_trajectories import load_data


def get_slews(actions):
    action_change = actions[:, :, 1:] - actions[:, :, :-1]
    action_change = action_change ** 2
    action_change = action_change.mean(dim=3)
    return action_change


env_str = 'drone'
load_dir = "logs/drone/predictor"
policy_type = 'random'
low_data = False
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
        extension = '_low_data' if alg_type != 'NeuralODE' and low_data else ''
        print(f"Extension = {extension}")
        states = torch.load(os.path.join(alg_dir, f'mpc_states{extension}.pt'))
        actions = torch.load(os.path.join(alg_dir, f'mpc_actions{extension}.pt'))
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
fig, ax = plt.subplots(figsize=(10, 10))
for alg_type, returns in results.items():
    # sum over all timesteps
    returns = returns[:, :, :100]
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
    ax.plot(xs, medians, label=alg_type)
    ax.fill_between(xs, mins, maxs, alpha=0.2)

    # ax.plot(xs, means, label=alg_type)
    # ax.fill_between(xs, [m-s for m,s in zip(means, stds)], [m+s for m,s in zip(means, stds)], alpha=0.2)

# ax.set_ylim(-100, 0)
ax.legend()
ax.set_xlabel('M')
ax.set_ylabel('Average Slew Rate')
plt.savefig(f'{load_dir}/mpc_slew_results.png')


# find largest mass index
max_mass = max([x['M'] for x in test_hidden_params])
max_mass_index = [i for i, x in enumerate(test_hidden_params) if x['M'] == max_mass][0]
# print(max_mass_index, max_mass)

# now look at an individual trajectory
print("\n\nPlotting trajectory")
env_index = 12 # 20
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
fig, ax = plt.subplots(figsize=(10, 10))
for alg_type, states in results.items():
    ax.plot(states[:, 4], label=alg_type)
# plot dotted line at z=1 for goal location
ax.plot([0, 100], [1, 1], 'k--', label='Goal')
ax.legend(loc="lower right")
ax.set_xlabel('Time')
ax.set_ylabel('Z (height)')
ax.set_title(f"Mass={mass:0.3f}")
plt.savefig(f'{load_dir}/mpc_z_results.png')