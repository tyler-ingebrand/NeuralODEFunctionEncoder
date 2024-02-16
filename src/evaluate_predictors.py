import os
import torch
import numpy as np
from tqdm import trange

from gather_trajectories import load_data


seed = 0
device = "cuda:0"
ignore_actions = False
env = "HalfCheetah-v3"
data_type = "random"
paths = [
"logs/cheetah/predictor/2024-02-13_16-27-03/MLP/random",
# "logs/cheetah/predictor/2024-02-13_16-30-59/NeuralODE/random",
# "logs/cheetah/predictor/2024-02-13_16-37-29/FE/random",
# "logs/cheetah/predictor/2024-02-13_16-43-20/FE_NeuralODE/random",
# "logs/cheetah/predictor/2024-02-13_18-06-20/FE_Residuals/random",
# "logs/cheetah/predictor/2024-02-13_18-14-21/FE_NeuralODE_Residuals/random",
]
# make sure they exist
for path in paths:
    assert os.path.exists(path), f"Path '{path}' does not exist. Make sure this path exists on your machine. You probably have to change the date-time. "


use_actions = not ignore_actions
if not use_actions:
    print("Ignoring actions. This will greatly reduce the performance for any data setting except on-policy.")
    print("For on-policy, it will only slightly decrease performance. ")

# seed everything
torch.manual_seed(seed)
np.random.seed(seed)

# load the data
_, _, _, test_states, test_actions, test_next_states = load_data(env, policy_type=data_type)
state_size, action_size = test_states.shape[-1], test_actions.shape[-1]

# get all predictors
predictors = []
for data_path in paths:
    alg_type = data_path.split("/")[-2]
    this_data_type = data_path.split("/")[-1]
    assert this_data_type == data_type
    assert alg_type in ["MLP", "NeuralODE", "FE", "FE_NeuralODE", "FE_Residuals", "FE_NeuralODE_Residuals"]

    # create predictor
    if alg_type == "MLP":
        from Predictors.MLP import MLP
        predictor = MLP(state_size, action_size, use_actions=use_actions).to(device)
    elif alg_type == "FE":
        from Predictors.FE import FE
        predictor = FE(state_size, action_size, use_actions=use_actions).to(device)
    elif alg_type == "NeuralODE":
        from Predictors.NeuralODE import NeuralODE
        predictor = NeuralODE(state_size, action_size, use_actions=use_actions).to(device)
    elif alg_type == "FE_NeuralODE":
        from Predictors.FE_NeuralODE import FE_NeuralODE
        predictor = FE_NeuralODE(state_size, action_size, use_actions=use_actions).to(device)
    elif alg_type == "FE_Residuals":
        from Predictors.FE import FE
        predictor = FE(state_size, action_size, use_actions=use_actions).to(device)
    elif alg_type == "FE_NeuralODE_Residuals":
        from Predictors.FE_NeuralODE_Residuals import FE_NeuralODE_Residuals
        predictor = FE_NeuralODE_Residuals(state_size, action_size, use_actions=use_actions).to(device)
    else:
        raise Exception("Unknown predictor")

    # load the model
    predictor.load_state_dict(torch.load(os.path.join(data_path, "model.pt")))
    predictor_dict = {"predictor": predictor, "alg_type": alg_type, "data_type": this_data_type}
    predictors.append(predictor_dict)

    # test to verify loading was successful
    raise Exception


# create the test sets. Needs an initial state, a sequence of actions, and the typical example states, actions, and next_states for function encoders
time_horizon = 100
number_samples = 200
num_trajectories = test_states.shape[1]
trajectory_lengths = test_states.shape[2]

# get test trajcetories
init_state_index = 0
initial_states = test_states[:, :, init_state_index, :]
real_next_states = test_states[:, :, init_state_index + 1:init_state_index + time_horizon+1, :]
action_sequences = test_actions[:, :, init_state_index:init_state_index + time_horizon, :]

# sample random example states, actions, and next_states
permutation = torch.randperm(num_trajectories * trajectory_lengths)
example_indices = permutation[:number_samples]
example_trajectory_indicies = example_indices // trajectory_lengths
example_timestep_indicies = example_indices % trajectory_lengths
example_states = test_states[:, example_trajectory_indicies, example_timestep_indicies, :]
example_actions = test_actions[:, example_trajectory_indicies, example_timestep_indicies, :]
example_next_states = test_next_states[:, example_trajectory_indicies, example_timestep_indicies, :]

# send everything to device
initial_states = initial_states.to(device)
action_sequences = action_sequences.to(device)
example_states = example_states.to(device)
example_actions = example_actions.to(device)
example_next_states = example_next_states.to(device)
real_next_states = real_next_states.to(device)

# evaluate the predictors
mses = {}
for predictor_dict in predictors:
    predictor = predictor_dict["predictor"]
    alg_type = predictor_dict["alg_type"]
    data_type = predictor_dict["data_type"]

    # predict the next states
    future_state_predictions = predictor.predict_trajectory(initial_states, action_sequences, example_states, example_actions, example_next_states)
    assert future_state_predictions.shape == real_next_states.shape, f"Expected shape {real_next_states.shape}, got {future_state_predictions.shape}"

    # compute time wise MSE
    mse = torch.mean((future_state_predictions - real_next_states)**2, dim=(0, 1, 3))
    mses[alg_type] = mse

# plot using matplotlib
import matplotlib.pyplot as plt
plt.figure()
for alg_type, mse in mses.items():
    plt.plot(mse.cpu().detach().numpy(), label=alg_type)
plt.legend()
plt.title("MSE over time horizon")
plt.xlabel("Time step")
plt.ylabel("MSE")
plt.show()