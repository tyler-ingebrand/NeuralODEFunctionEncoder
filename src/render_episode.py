import os
import time

import cv2
import numpy as np
import torch
from tqdm import trange

from gather_trajectories import load_data


hidden_param_index = 2
trajectory_index = 0

# script parameters
env_str = 'drone' # "Ant-v3" # "HalfCheetah-v3" # 'drone
policy_type = "random"
datetime_str = "2024-03-08_23-09-07"
alg_type = "FE_NeuralODE_Residuals"
# datetime_str = "2024-03-08_21-29-13"
# alg_type = "MLP"
use_actions = True
normalize = True


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

with torch.no_grad():
    # load data
    _, _, _, states, actions, next_states, train_hidden_params, hidden_params = load_data(env_str, policy_type, normalize=False, get_groundtruth_hidden_params=True)
    _, _, _, _, _, _, mean, std = load_data(env_str, policy_type, normalize=True, get_normalize_params=True)

    print("Means: ", mean)
    print("Stds: ", std)

    # create the env
    hidden_params = hidden_params[hidden_param_index]
    hidden_params_dict = {key: (value, value) for key, value in hidden_params.items()}
    if env_str == "HalfCheetah-v3":
        from VariableCheetahEnv import VariableCheetahEnv
        env = VariableCheetahEnv(hidden_params_dict, render_mode="rgb_array")
    elif env_str == "Ant-v3":
        from VariableAntEnv import VariableAntEnv
        env = VariableAntEnv(hidden_params_dict, render_mode="rgb_array")
    elif env_str == "drone":
        from VariableDroneEnvironment import VariableDroneEnv
        env = VariableDroneEnv(hidden_params_dict, render_mode="rgb_array")
    else:
        raise ValueError(f"Unknown env '{env_str}'")
    _ = env.reset()
    img = env.render()
    width, height = img.shape[1], img.shape[0]
    state_shape, action_shape = states.shape[-1], actions.shape[-1]

    # load a predictor
    alg_dir = f"logs/{'ant' if env_str == 'Ant-v3' else 'cheetah' if env_str == 'HalfCheetah-v3' else 'drone'}/predictor/{datetime_str}/{alg_type}/{policy_type}"
    if alg_type == "MLP":
        from Predictors.MLP import MLP
        predictor = MLP(state_shape, action_shape, use_actions=use_actions)
    elif alg_type == "FE":
        from Predictors.FE import FE
        predictor = FE(state_shape, action_shape, use_actions=use_actions)
    elif alg_type == "NeuralODE":
        from Predictors.NeuralODE import NeuralODE
        predictor = NeuralODE(state_shape, action_shape, use_actions=use_actions)
    elif alg_type == "FE_NeuralODE":
        from Predictors.FE_NeuralODE import FE_NeuralODE
        predictor = FE_NeuralODE(state_shape, action_shape, use_actions=use_actions)
    elif alg_type == "FE_Residuals":
        from Predictors.FE import FE
        predictor = FE(state_shape, action_shape, use_actions=use_actions)
    elif alg_type == "FE_NeuralODE_Residuals":
        from Predictors.FE_NeuralODE_Residuals import FE_NeuralODE_Residuals
        predictor = FE_NeuralODE_Residuals(state_shape, action_shape, use_actions=use_actions)
    elif alg_type == "Oracle":
        from Predictors.Oracle import Oracle
        min_vals_hps = {key: min([x[key] for x in train_hidden_params]) for key in train_hidden_params[0].keys()}
        max_vals_hps = {key: max([x[key] for x in train_hidden_params]) for key in train_hidden_params[0].keys()}
        predictor = Oracle(state_shape, action_shape, min_vals_hps=min_vals_hps, max_vals_hps=max_vals_hps, use_actions=use_actions)
        del train_hidden_params
    else:
        raise Exception("Unknown predictor")
    predictor.load_state_dict(torch.load(os.path.join(alg_dir, "model.pt")))


    # create a mp4 via cv
    width, height, fps = width * 2, height, 10
    out = cv2.VideoWriter(f'{alg_dir}/approx_trajectory.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # get example data points
    num_examples = 200
    perm = torch.randperm(states.shape[1] * states.shape[2])
    example_indicies = perm[:num_examples]
    example_episode_indicies = example_indicies // states.shape[2]
    example_timestep_indicies = example_indicies % states.shape[2]
    example_states = states[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]
    example_actions = actions[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]
    example_next_states = next_states[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]

    # normalize example data
    if normalize:
        example_states = (example_states - mean) / std
        example_next_states = (example_next_states - mean) / std


    # render an episode
    trajectory = states[hidden_param_index, trajectory_index, :100, :]
    trajectory_actions = actions[hidden_param_index, trajectory_index, :100, :]
    current_state_approx = trajectory[0, :]

    if normalize:
        normalized_current_state_approx = (current_state_approx - mean) / std
    else:
        normalized_current_state_approx = current_state_approx



    for time_index in trange(trajectory.shape[0]):
        # set true env rendering
        true_state = trajectory[time_index, :]
        if env_str == "HalfCheetah-v3":
            qpos = true_state[:9]
            qvel = true_state[9:]
            env.env.set_state(qpos, qvel)
        elif env_str == "Ant-v3":
            qpos = true_state[:15]
            qvel = true_state[15:]
            env.env.set_state(qpos, qvel)
        else:
            env.set_state(true_state)

        # render image
        true_img = env.render()

        # set approx env rendering
        if normalize:
            current_state_approx = normalized_current_state_approx * std + mean
        else:
            current_state_approx = normalized_current_state_approx
        if env_str == "HalfCheetah-v3":
            qpos = current_state_approx[:9].cpu().numpy()
            qvel = current_state_approx[9:].cpu().numpy()
            env.env.set_state(qpos, qvel)
        elif env_str == "Ant-v3":
            qpos = current_state_approx[:15].cpu().numpy()
            qvel = current_state_approx[15:].cpu().numpy()
            env.env.set_state(qpos, qvel)
        else:
            env.set_state(current_state_approx.cpu().numpy())

        # render image
        approx_img = env.render()

        # calculate loss
        if normalize:
            normalized_true_state = (true_state - mean) / std
            loss = torch.nn.functional.mse_loss(normalized_current_state_approx, normalized_true_state)
        else:
            loss = torch.nn.functional.mse_loss(current_state_approx, true_state)

        # calculate approx next state
        action = trajectory_actions[time_index, :]
        if alg_type == "Oracle":
            normalized_current_state_approx, _ = predictor.predict(normalized_current_state_approx.unsqueeze(0).unsqueeze(0), action.unsqueeze(0).unsqueeze(0), hidden_params=[hidden_params[hidden_param_index]])
        else:
            normalized_current_state_approx, _ = predictor.predict(normalized_current_state_approx.unsqueeze(0).unsqueeze(0), action.unsqueeze(0).unsqueeze(0), example_states.unsqueeze(0), example_actions.unsqueeze(0), example_next_states.unsqueeze(0))
        normalized_current_state_approx = normalized_current_state_approx.squeeze(0).squeeze(0)

        # create an image, write loss to top corner, write labels to bottom corner, write time to the top corner
        img = np.concatenate((true_img, approx_img), axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.putText(img, f"Loss: {loss.item():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, f"Ground Truth", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, f"Approximate", (width // 2 + 10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, f"Time: {time_index}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # write to mp4
        out.write(img)



    env.close()
    out.release()