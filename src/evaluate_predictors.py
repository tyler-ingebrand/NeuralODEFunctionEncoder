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
argparser.add_argument("--env", type=str, default="HalfCheetah-v3")
argparser.add_argument("--data_type", type=str, default="random")
argparser.add_argument("--normalize", action="store_true")
argparser.add_argument("--load_mses", action="store_true")
args = argparser.parse_args()

seed = args.seed
device = args.device
ignore_actions = args.ignore_actions
env = args.env
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
    # load the data
    _, _, _, test_states, test_actions, test_next_states, train_hidden_params, test_hidden_params = load_data(env, policy_type=data_type, normalize=normalize, get_groundtruth_hidden_params=True)
    state_size, action_size = test_states.shape[-1], test_actions.shape[-1]

    # get all predictors
    print("The following losses were observed after testing the loaded models on the test set. ")
    print("Make sure they are very close to the error observed during training. If not, there may be an error in loading. ")
    predictors = []
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

        # test to verify loading was successful
        # now do the same thing for testing only
        total_test_loss = 0.0
        grad_accumulation_steps = 5
        n_envs_at_once = 10
        n_example_points = 200
        # device = "cuda:0"
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
            if predictor_dict["alg_type"] == "Oracle":
                predicted_next_states, test_info = predictor.predict(states, actions, [test_hidden_params[i] for i in env_indicies])
            else:
                predicted_next_states, test_info = predictor.predict(states, actions, example_states, example_actions, example_next_states)
            test_loss = torch.nn.functional.mse_loss(predicted_next_states, next_states)
            total_test_loss += test_loss.item() / grad_accumulation_steps
            del states, actions, next_states, example_states, example_actions, example_next_states, predicted_next_states, test_loss
        print(f"{alg_type}, loss: {total_test_loss:2f}")


    # load the zeros predictor, which only predicts the average state from the dataset
    # from Predictors.Zeros import Zeros
    # predictor_dict = {"predictor": Zeros(state_size=predictors[0]["predictor"].state_size,
    #                                      action_size=predictors[0]["predictor"].action_size),
    #                   "alg_type": "Zeros",
    #                   "data_type": this_data_type}
    # predictors.append(predictor_dict)

    # create the test sets. Needs an initial state, a sequence of actions, and the typical example states, actions, and next_states for function encoders
    time_horizon = 100
    number_samples = 200
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
            mses[alg_type].append(test_mses)


    # convert every mse to a tensor, of size num_trials x num_envs x num_trajs x time_horizon
    for alg_type, mse in mses.items():
        mses[alg_type] = torch.stack(mse, dim=0)

    # save the mses
    torch.save(mses, os.path.join(save_dir, "mses.pt"))
else:
    mses = torch.load(os.path.join(save_dir, "mses.pt"))


# labels for every alg
label_dict = {"MLP": "MLP", 
              "FE": "FE", 
              "NeuralODE": "NeuralODE",
              "FE_NeuralODE": "FE + NeuralODE", 
              "FE_Residuals": "FE + Recentering", 
              "FE_NeuralODE_Residuals": "FE + NeuralODE + Recentering", 
              "Oracle": "Oracle"}

# for every alg type, plot the mse over the time horizon. Compute the quartiles for plotting
fig = plt.figure()
for alg_type, mse in mses.items():
    print(alg_type, ": ", mse.shape[0], " seeds")
    # time_mse = torch.mean(mse, dim=(1,2))
    time_mse = mse.reshape(-1, mse.shape[-1])
    quartiles = np.percentile(time_mse.cpu().detach().numpy(), [25, 50, 75], axis=0)
    label = label_dict[alg_type]
    plt.plot(quartiles[1], label=label)
    # plt.fill_between(range(quartiles.shape[1]), quartiles[0], quartiles[2], alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("MSE over time horizon")
plt.xlabel("Time step")
plt.ylabel("MSE")
y_lim, x_lim = None, None
if normalize:
    if env == "HalfCheetah-v3":
        y_lim = 1.0
        x_lim = 99
    elif env == "drone":
        y_lim = 1.0
        x_lim = 30
    else:
        y_lim = 3.0
        x_lim = 99
if not normalize:
    y_lim = 15.0
    x_lim = 99
plt.ylim(0, y_lim)
plt.xlim(0, x_lim)
plt.savefig(os.path.join(save_dir , "mse_over_time_horizon.png"), bbox_inches="tight", dpi=500)
plt.clf()

# now do the same thing, but filter "hard" examples
# remove any trajectory with a mse over mse_limit
mse_limit = 1000
mses_reshape = {alg_type: mse.reshape(mse.shape[0], -1, mse.shape[-1]) for alg_type, mse in mses.items()}
max_mses = {alg_type: torch.max(mses, dim=2).values for alg_type, mses in mses_reshape.items()}
max_mses = {alg_type: torch.max(max_mses, dim=0).values for alg_type, max_mses in max_mses.items()}
to_removes = {alg_type: max_mses > mse_limit for alg_type, max_mses in max_mses.items()}
to_remove = torch.zeros_like(list(to_removes.values())[0])
for alg_type, to_remove_alg in to_removes.items():
    to_remove = to_remove_alg | to_remove
filtered_mses = {alg_type: mse[:, ~to_remove, :] for alg_type, mse in mses_reshape.items()}

# now plot
fig = plt.figure()
for alg_type, mse in filtered_mses.items():
    # time_mse = torch.mean(mse, dim=1)
    time_mse = mse.reshape(-1, mse.shape[-1])
    quartiles = np.percentile(time_mse.cpu().detach().numpy(), [25, 50, 75], axis=0)
    plt.plot(quartiles[1], label=alg_type)
    plt.fill_between(range(quartiles.shape[1]), quartiles[0], quartiles[2], alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("MSE over time horizon")
plt.xlabel("Time step")
plt.ylabel("MSE")
plt.ylim(0, 1.0 if normalize else 15.0)
plt.savefig(os.path.join(save_dir , "filtered_mse_over_time_horizon.png"), bbox_inches="tight")


