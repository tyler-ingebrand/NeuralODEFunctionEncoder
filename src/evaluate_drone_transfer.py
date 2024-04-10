import argparse
import os
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from gather_trajectories import load_data


argparser = argparse.ArgumentParser()
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--ignore_actions", action="store_true")
argparser.add_argument("--data_type", type=str, default="random")
argparser.add_argument("--normalize", action="store_true")
argparser.add_argument("--load_mses", action="store_true")
args = argparser.parse_args()

seed = args.seed
device = args.device
ignore_actions = args.ignore_actions
env = 'drone'
data_type = args.data_type
normalize = args.normalize
load_mses = args.load_mses



# find all paths in logs/cheetah/predictor
subdirs = next(os.walk(f"logs/{'cheetah' if env == 'HalfCheetah-v3' else 'ant' if env == 'Ant-v3' else 'drone'}/predictor"))[1]
subdirs = [os.path.join(f"logs/{'cheetah' if env == 'HalfCheetah-v3' else 'ant' if env == 'Ant-v3' else 'drone'}/predictor", x) for x in subdirs]
# every subdir has a policy in it, add this extension
sub_sub_dirs = []
for subdir in subdirs:
    sub_sub_dirs.append(os.path.join(subdir, next(os.walk(subdir))[1][0]))
sub_sub_sub_dirs = []
for subdir in sub_sub_dirs: # now adds the datatype to it
    next_dir = next(os.walk(subdir))[1][0]
    if next_dir == data_type:
        sub_sub_sub_dirs.append(os.path.join(subdir, next_dir))
paths = sub_sub_sub_dirs
# paths = [x for x in paths if "Oracle" in x][0]


# where to save file
save_dir = os.path.join("logs", "cheetah" if env == "HalfCheetah-v3" else "ant" if env == 'Ant-v3' else 'drone', "predictor")

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

if not load_mses:
    with torch.no_grad():
        # load the data
        _, _, _, test_states, test_actions, test_next_states, train_hidden_params, test_hidden_params = load_data(env, policy_type=data_type, normalize=normalize, get_groundtruth_hidden_params=True)
        state_size, action_size = test_states.shape[-1], test_actions.shape[-1]

        # get all predictors
        predictors = []
        predictor_losses = {}
        for data_path in paths:
            alg_type = data_path.split("/")[-2]
            this_data_type = data_path.split("/")[-1]
            if this_data_type != data_type:
                print(f"Skipping {data_path} because it is not of the data type {data_type}")
                continue
            if alg_type not in ["MLP", "NeuralODE", "FE", "FE_NeuralODE", "FE_Residuals", "FE_NeuralODE_Residuals", "Oracle"]:
                print(f"Skipping {data_path} because it is not a known predictor")
                continue

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
            elif alg_type == "Oracle":
                from Predictors.Oracle import Oracle
                min_vals_hps = {key: min([x[key] for x in train_hidden_params]) for key in train_hidden_params[0].keys()}
                max_vals_hps = {key: max([x[key] for x in train_hidden_params]) for key in train_hidden_params[0].keys()}
                predictor = Oracle(state_size, action_size, min_vals_hps=min_vals_hps, max_vals_hps=max_vals_hps, use_actions=use_actions).to(device)
            else:
                raise Exception("Unknown predictor")

            # load the model
            try:
                predictor.load_state_dict(torch.load(os.path.join(data_path, "model.pt")))
            except Exception as e:
                print(f"Failed to load model from {data_path}. Error: {e}")
                continue
            predictor_dict = {"predictor": predictor, "alg_type": alg_type, "data_type": this_data_type}
            predictors.append(predictor_dict)
            print("Testing ", alg_type, end="")

            # test to verify loading was successful
            # now do the same thing for testing only
            losses = []
            n_example_points = 2000
            max_test_at_once = 48000
            # device = "cuda:0"
            for env_index in range(test_states.shape[0]):
                print(".", end="")

                # get some random steps
                perm = torch.randperm(test_states.shape[1] * test_states.shape[2])
                example_indicies = perm[:n_example_points]
                train_indicies_all = perm[n_example_points:]  # only gather the first 800 random points

                # convert to episode and timestep indicies
                assert train_indicies_all.shape[0] % max_test_at_once == 0, "This code assumes that the number of test indicies is divisible by max_test_at_once"
                num_batches = train_indicies_all.shape[0] // max_test_at_once
                total_loss = 0
                for b in range(num_batches):
                    train_indicies = train_indicies_all[b*max_test_at_once:(b+1)*max_test_at_once]
                    example_episode_indicies = example_indicies // test_states.shape[2]
                    example_timestep_indicies = example_indicies % test_states.shape[2]
                    train_episode_indicies = train_indicies // test_states.shape[2]
                    train_timestep_indicies = train_indicies % test_states.shape[2]

                    # gather data
                    states = test_states[env_index:env_index+1, :, :, :][:, train_episode_indicies, train_timestep_indicies,:].to(device)
                    actions = test_actions[env_index:env_index+1, :, :, :][:, train_episode_indicies, train_timestep_indicies, :].to(device)
                    next_states = test_next_states[env_index:env_index+1, :, :, :][:, train_episode_indicies, train_timestep_indicies, :].to(device)
                    example_states = test_states[env_index:env_index+1, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(device)
                    example_actions = test_actions[env_index:env_index+1, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(device)
                    example_next_states = test_next_states[env_index:env_index+1, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(device)

                    # reshape to ignore the episode dim, since we are only doing 1 step, it doesnt matter
                    states = states.view(states.shape[0], -1, states.shape[-1])
                    actions = actions.view(actions.shape[0], -1, actions.shape[-1])
                    next_states = next_states.view(next_states.shape[0], -1, next_states.shape[-1])
                    example_states = example_states.view(example_states.shape[0], -1, example_states.shape[-1])
                    example_actions = example_actions.view(example_actions.shape[0], -1, example_actions.shape[-1])
                    example_next_states = example_next_states.view(example_next_states.shape[0], -1, example_next_states.shape[-1])

                    # test the predictor
                    if predictor_dict["alg_type"] == "Oracle":
                        predicted_next_states, test_info = predictor.predict(states, actions, [test_hidden_params[env_index]])
                    else:
                        predicted_next_states, test_info = predictor.predict(states, actions, example_states, example_actions, example_next_states)
                    test_loss = torch.nn.functional.mse_loss(predicted_next_states, next_states)
                total_loss += test_loss.item() / num_batches
                losses.append((test_hidden_params[env_index], total_loss))
                del states, actions, next_states, example_states, example_actions, example_next_states, predicted_next_states, test_loss, total_loss
            predictor_losses[alg_type] = losses
            print("")



        # create the test sets. Needs an initial state, a sequence of actions, and the typical example states, actions, and next_states for function encoders
        time_horizon = 10
        number_samples = 2000
        num_trajectories = test_states.shape[1]
        trajectory_lengths = test_states.shape[2]

        # get test trajcetories
        init_state_index = 20
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
                if alg_type == "Oracle":
                    future_state_predictions = predictor.predict_trajectory(initial_states, action_sequences, test_hidden_params)
                else:
                    future_state_predictions = predictor.predict_trajectory(initial_states, action_sequences, example_states, example_actions, example_next_states)
                assert future_state_predictions.shape == real_next_states.shape, f"Expected shape {real_next_states.shape}, got {future_state_predictions.shape}"

                # compute time wise MSE. Gets mse for every env, every trajectory, and every time step
                test_mses = torch.mean((future_state_predictions - real_next_states)**2, dim=3)

                if not mses.get(alg_type):
                    mses[alg_type] = []
                mses[alg_type].append((test_hidden_params, test_mses))


        # convert every mse to a tensor, of size num_trials x num_envs x num_trajs x time_horizon
        # for alg_type, mse in mses.items():
        #     mses[alg_type] = torch.stack(mse, dim=0)

        # save the mses
        torch.save(mses, os.path.join(save_dir, "drone_specific_k_step_mses.pt"))
        torch.save(predictor_losses, os.path.join(save_dir, "drone_specific_predictor_losses.pt"))
else:
    mses = torch.load(os.path.join(save_dir, "drone_specific_k_step_mses.pt"))
    predictor_losses = torch.load(os.path.join(save_dir, "drone_specific_predictor_losses.pt"))


# labels for every alg
label_dict = {"MLP": "MLP", 
              "FE": "FE", 
              "NeuralODE": "NeuralODE",
              "FE_NeuralODE": "FE + NeuralODE", 
              "FE_Residuals": "FE + Recentering", 
              "FE_NeuralODE_Residuals": "FE + NeuralODE + Recentering", 
              "Oracle": "Oracle"}

# First plot the MSE for each env, with mass on the X axis and MSE on the Y axis
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for alg_type in predictor_losses:
    if alg_type == "FE_Residuals":
        print("Skipping")
        continue
    alg_losses = predictor_losses[alg_type]
    alg_losses_sorted = sorted(alg_losses, key=lambda x: x[0]["M"])

    # now plot these sorted values
    x = [x[0]["M"] for x in alg_losses_sorted]
    y = [x[1] for x in alg_losses_sorted]
    ax.plot(x, y, label=label_dict[alg_type])

ax.set_xlabel("Mass")
ax.set_ylabel("MSE")
ax.set_title("MSE vs Mass")
ax.legend()
plt.savefig(os.path.join(save_dir, "drone_specific_mse_vs_mass.png"))


# second plot the mse for a k-step prediction, with mass on the X axis and MSE at the end on the Y axis
plt.clf()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
k = -1
for alg_type in mses:
    if alg_type == "FE_Residuals":
        print("Skipping")
        continue
    all_seeds = mses[alg_type]
    for seed_mses in all_seeds:
        hps, mse = seed_mses
        k = mse.shape[-1]
        mse = mse[:, :, -1]
        mse = mse.mean(dim=-1) # mean over all trajectories. Leaves each env alone
        assert len(hps) == mse.shape[0]
        hp_mse = [(x, y) for x, y in zip(hps, mse)]
        hp_mse = sorted(hp_mse, key=lambda x: x[0]["M"])
        x = [x[0]["M"] for x in hp_mse]
        y = [x[1].item() for x in hp_mse]
        ax.plot(x, y, label=label_dict[alg_type])

ax.set_xlabel("Mass")
ax.set_ylabel(f"{k}-step MSE")
ax.set_title("MSE vs Mass")
ax.legend()
plt.savefig(os.path.join(save_dir, "drone_specific_k_step_mse_vs_mass.png"))