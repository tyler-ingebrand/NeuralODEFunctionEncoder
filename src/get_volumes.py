import torch
import argparse
import numpy as np
from gather_trajectories import load_data
from tqdm import trange

argparser = argparse.ArgumentParser()
argparser.add_argument("--env", type=str, required=True)
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--device", type=str, default="cuda:0")
args = argparser.parse_args()

# seed everything
torch.manual_seed(args.seed)
np.random.seed(args.seed)

data_types = ['random', 'on_policy', "precise", "precise2"]

# get consistent normalization
states, actions, _ = load_data(args.env, "random", normalize=False)
states = states.reshape(-1, states.shape[-1])
mean, std = states.mean(dim=0), states.std(dim=0)

# set ranges for sampling
lower = torch.concat((torch.ones(states.shape[-1]) * -3, torch.ones(actions.shape[-1]) * -1), dim=-1)
upper = -lower


results = {}
num_sampling_points = int(1e4)
min_distance = 5
for sampling_type in data_types:
    states, actions, _ = load_data(args.env, sampling_type, normalize=False)
    states = states.reshape(-1, states.shape[-1])
    actions = actions.reshape(-1, actions.shape[-1])

    # normalize
    states = (states - mean) / std # noramlize so its consistent
    data = torch.concat((states, actions), dim=-1)

    # sample random points within the range
    accepted_points = 0
    for _ in trange(num_sampling_points):
        sample = torch.rand(data.shape[-1]) * (upper - lower) + lower
        distances = (data - sample).norm(dim=-1)
        if (distances < min_distance).any():
            accepted_points += 1

    percentage = accepted_points / num_sampling_points
    results[sampling_type] = percentage

for key, val in results.items():
    print(f"{key}: {val}")


