import os
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from gather_trajectories import load_data


seed = 0
device = "cuda:0"
ignore_actions = False
env = "HalfCheetah-v3"
data_type = "random"
paths = [
"logs/cheetah/predictor/normalized/2024-02-20_11-33-30/MLP/random",
"logs/cheetah/predictor/normalized/2024-02-20_11-38-19/NeuralODE/random",
"logs/cheetah/predictor/normalized/2024-02-20_11-45-53/FE/random",
"logs/cheetah/predictor/normalized/2024-02-20_11-52-22/FE_NeuralODE/random",
"logs/cheetah/predictor/normalized/2024-02-20_13-17-49/FE_Residuals/random",
"logs/cheetah/predictor/normalized/2024-02-20_13-26-29/FE_NeuralODE_Residuals/random",
]
save_dir = os.path.join("logs", "cheetah" if env == "HalfCheetah-v3" else "ant", "predictor")
normalize = True

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
_, _, _, test_states, test_actions, test_next_states = load_data(env, policy_type=data_type, normalize=normalize)
state_size, action_size = test_states.shape[-1], test_actions.shape[-1]

# get all predictors
print("The following losses were observed after testing the loaded models on the test set. ")
print("Make sure they are very close to the error observed during training. If not, there may be an error in loading. ")
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
    # now do the same thing for testing only
    total_test_loss = 0.0
    grad_accumulation_steps = 5
    n_envs_at_once = 10
    n_example_points = 200
    device = "cuda:0"
    for grad_accum_step in range(grad_accumulation_steps):
        # randomize
        # get some envs
        env_indicies = torch.randperm(test_states.shape[0])[:n_envs_at_once]

        # get some random steps
        perm = torch.randperm(test_states.shape[1] * test_states.shape[2])
        example_indicies = perm[:n_example_points]
        train_indicies = perm[n_example_points:][:800]  # only gather the first 800 random points

        # convert to episode and timestep indicies
        example_episode_indicies = example_indicies // test_states.shape[2]
        example_timestep_indicies = example_indicies % test_states.shape[2]
        train_episode_indicies = train_indicies // test_states.shape[2]
        train_timestep_indicies = train_indicies % test_states.shape[2]

        # gather data
        states = test_states[env_indicies, :, :, :][:, train_episode_indicies, train_timestep_indicies,:].to(device)
        actions = test_actions[env_indicies, :, :, :][:, train_episode_indicies, train_timestep_indicies, :].to(device)
        next_states = test_next_states[env_indicies, :, :, :][:, train_episode_indicies, train_timestep_indicies, :].to(device)
        example_states = test_states[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(device)
        example_actions = test_actions[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(device)
        example_next_states = test_next_states[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(device)

        # reshape to ignore the episode dim, since we are only doing 1 step, it doesnt matter
        states = states.view(states.shape[0], -1, states.shape[-1])
        actions = actions.view(actions.shape[0], -1, actions.shape[-1])
        next_states = next_states.view(next_states.shape[0], -1, next_states.shape[-1])
        example_states = example_states.view(example_states.shape[0], -1, example_states.shape[-1])
        example_actions = example_actions.view(example_actions.shape[0], -1, example_actions.shape[-1])
        example_next_states = example_next_states.view(example_next_states.shape[0], -1, example_next_states.shape[-1])

        # test the predictor
        predicted_next_states, test_info = predictor.predict(states, actions, example_states, example_actions, example_next_states)
        test_loss = torch.nn.functional.mse_loss(predicted_next_states, next_states)
        total_test_loss += test_loss.item() / grad_accumulation_steps
        del states, actions, next_states, example_states, example_actions, example_next_states, predicted_next_states, test_loss
    print(f"{alg_type}, loss: {total_test_loss:2f}")


# load the zeros predictor, which only predicts the average state from the dataset
from Predictors.Zeros import Zeros
predictor_dict = {"predictor": Zeros(state_size=predictors[0]["predictor"].state_size,
                                     action_size=predictors[0]["predictor"].action_size),
                  "alg_type": "Zeros",
                  "data_type": this_data_type}
predictors.append(predictor_dict)

# create the test sets. Needs an initial state, a sequence of actions, and the typical example states, actions, and next_states for function encoders
time_horizon = 100
number_samples = 200
num_trajectories = test_states.shape[1]
trajectory_lengths = test_states.shape[2]

# get test trajcetories
init_state_index = 5
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
print("")
mses = {}
for predictor_dict in predictors:
    predictor = predictor_dict["predictor"]
    alg_type = predictor_dict["alg_type"]
    data_type = predictor_dict["data_type"]
    print("Testing ", alg_type)

    # predict the next states
    with torch.no_grad():
        future_state_predictions = predictor.predict_trajectory(initial_states, action_sequences, example_states, example_actions, example_next_states)
        assert future_state_predictions.shape == real_next_states.shape, f"Expected shape {real_next_states.shape}, got {future_state_predictions.shape}"

        # compute time wise MSE
        test_mses = torch.mean((future_state_predictions - real_next_states)**2, dim=3)
        torch.save(test_mses, os.path.join("/tmp", f"{alg_type}_mse.pt"))

        mse = torch.mean((future_state_predictions - real_next_states)**2, dim=(0, 1, 3))
        mses[alg_type] = mse

# plot using matplotlib
plt.figure()
for alg_type, mse in mses.items():
    plt.plot(mse.cpu().detach().numpy(), label=alg_type)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.legend()
plt.title("MSE over time horizon")
plt.xlabel("Time step")
plt.ylabel("MSE")
plt.ylim(0, 1.0 if normalize else 15.0)
# plt.show()
plt.savefig(os.path.join(save_dir , "mse_over_time_horizon.png"), bbox_inches="tight")