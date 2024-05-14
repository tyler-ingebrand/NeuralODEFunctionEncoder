import argparse

import numpy as np
import torch
from torch import jit
from torch import nn, optim
import os
import cv2
from tqdm import trange, tqdm
import colorednoise as cn
import math

from gather_trajectories import load_data
from src.CEM_MPC import CEM
from src.Grad_CEM_MPC import GradCEM
from src.Grad_MPC import Grad
from src.MPC_Env import MPCEnv
from VariableDroneEnvironment import VariableDroneEnv


def convert_quat_state_to_euler(quat_state):
    x, dx, y, dy, z, dz = quat_state[0], quat_state[1], quat_state[2], quat_state[3], quat_state[4], quat_state[5]
    quat1 = quat_state[6]
    quat2 = quat_state[7]
    quat3 = quat_state[8]
    quat4 = quat_state[9]
    p = quat_state[10]
    q = quat_state[11]
    r = quat_state[12]

    # convert quats eo euler
    # normalize quat
    mag = math.sqrt(quat1 ** 2 + quat2 ** 2 + quat3 ** 2 + quat4 ** 2)
    quat1 /= mag
    quat2 /= mag
    quat3 /= mag
    quat4 /= mag

    psi = math.atan2(2 * (quat1 * quat2 + quat3 * quat4), 1 - 2 * (quat2 ** 2 + quat3 ** 2))
    theta = -math.asin(2 * (quat1 * quat3 - quat4 * quat2))
    phi = math.atan2(2 * (quat1 * quat4 + quat2 * quat3), 1 - 2 * (quat3 ** 2 + quat4 ** 2))
    if phi > math.pi / 2:
        phi -= math.pi
    elif phi < -math.pi / 2:
        phi += math.pi
    phi = phi * -1
    return [x, dx, y, dy, z, dz, phi, theta, psi, p, q, r]


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--alg", type=str, default="NeuralODE")
    args = argparser.parse_args()
    only_this_alg = args.alg


    # hyper params
    test_horizon = 100
    planning_horizon = 10
    opt_iters = 100
    samples = 100 # MPC sample trajs to optimize at once
    num_episodes_per_hidden_param = 5
    render = False
    low_data = False

    env_str = 'drone'
    load_dir = "logs/drone/predictor"
    policy_type = 'random'

    use_actions = True
    normalize = True
    seed = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mpc params
    top_samples = int(0.4 * samples)
    mpc_version = "Grad"

    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # actual code
    # load data
    _, training_actions123, _, states, actions, next_states, train_hidden_params, test_hidden_params = load_data(env_str, policy_type, normalize=False, get_groundtruth_hidden_params=True)
    _, _, _, _, _, _, mean, std = load_data(env_str, policy_type, normalize=True, get_normalize_params=True)

    # compute state and action shapes
    state_shape = states.shape[-1]
    action_shape = actions.shape[-1]

    # calculate the action limits in the data
    training_actions123 = training_actions123.reshape(-1, training_actions123.shape[-1])
    action_means = training_actions123.mean(dim=0)
    quants = torch.quantile(training_actions123, torch.tensor([0.1, 0.9]), dim=0)
    action_space = torch.stack([quants[0], quants[1]], dim=1)
    del training_actions123

    # create initial state conditions based on past data. This is so the initial states are the same
    # for all algorithms. Otherwise, the results are noisy.
    initial_states = states[0, :num_episodes_per_hidden_param, 0, :].clone().detach()

    # fetch all the subdirectories
    alg_dirs = [os.path.join(load_dir, x) for x in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, x))]
    seed = 0

    for alg_dir in alg_dirs:
        # load the subdir. There should be only 1
        subdirs = [x for x in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, x))]
        assert len(subdirs) == 1, f"Expected 1 subdir, got {len(subdirs)}"
        alg_type = subdirs[0]
        alg_dir = os.path.join(alg_dir, alg_type, policy_type)


        # we want to skip FE for now
        if alg_type != only_this_alg:
            print("Skipping", alg_type)
            continue

        # load the model
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
            from Predictors.FE_NeuralODE_Fast import FE_NeuralODE_Fast
            predictor = FE_NeuralODE(state_shape, action_shape, use_actions=use_actions)
        elif alg_type == "FE_Residuals":
            from Predictors.FE import FE
            predictor = FE(state_shape, action_shape, use_actions=use_actions)
        elif alg_type == "FE_NeuralODE_Residuals":
            from Predictors.FE_NeuralODE_Residuals import FE_NeuralODE_Residuals
            from Predictors.FE_NeuralODE_Residuals_Fast import FE_NeuralODE_Residuals_Fast
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

        # efficient models for MPC
        if alg_type == "FE_NeuralODE":
            print("FAST")
            fast_predictor = FE_NeuralODE_Fast(predictor)
            del predictor
            predictor = fast_predictor
        elif alg_type == "FE_NeuralODE_Residuals":
            fast_predictor = FE_NeuralODE_Residuals_Fast(predictor)
            del predictor
            predictor = fast_predictor

        # verify performance
        # predictor = predictor.to(device)
        # torch.manual_seed(-10)
        # torch.cuda.manual_seed(-10)
        # with torch.no_grad():
        #     predictor.verify_performance(states, actions, next_states, mean, std, device=device, normalize=normalize)
        # exit(-1)

        # create a renderer
        if render:
            hidden_params = test_hidden_params[0]
            hidden_params_dict = {key: (value, value) for key, value in hidden_params.items()}
            env = VariableDroneEnv(hidden_params_dict, render_mode="rgb_array", seed=seed)
            img = env.render()
            width, height = img.shape[1], img.shape[0]
            fps = 10
            out = cv2.VideoWriter(f'{alg_dir}/mpc_trajectory.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            env.close()
            del env


        # create tensors to store states and actions for all envs.
        # this way we can save the trajectories and do whatever analysis later
        states_to_save = torch.zeros(len(test_hidden_params), num_episodes_per_hidden_param, test_horizon+1, state_shape, device="cpu")
        actions_to_save = torch.zeros(len(test_hidden_params), num_episodes_per_hidden_param, test_horizon, action_shape, device="cpu")

        tbar = tqdm(total=len(test_hidden_params) * num_episodes_per_hidden_param)
        tbar.set_description(f"{alg_type}")
        for hidden_param_index in range(len(test_hidden_params)):
            if render and hidden_param_index not in (12, 39):
                continue
            hidden_params = test_hidden_params[hidden_param_index]
            hidden_params_dict = {key: (value, value) for key, value in hidden_params.items()}

            seed += 1
            env = VariableDroneEnv(hidden_params_dict, render_mode="rgb_array", seed=seed)

            _ = env.reset(add_sphere=render)
            state_space = env.observation_space

            # get example data points
            num_examples = 2000 if not low_data else 200
            perm = torch.randperm(states.shape[1] * states.shape[2])
            example_indicies = perm[:num_examples]
            example_episode_indicies = example_indicies // states.shape[2]
            example_timestep_indicies = example_indicies % states.shape[2]
            example_states = states[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]
            example_actions = actions[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]
            example_next_states = next_states[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]

            # convert device of everything
            example_states = example_states.to(device)
            example_actions = example_actions.to(device)
            example_next_states = example_next_states.to(device)
            mean = mean.to(device)
            std = std.to(device)
            predictor = predictor.to(device)

            if normalize:
                example_states = (example_states - mean) / std
                example_next_states = (example_next_states - mean) / std

            # update the representation
            if alg_type == "FE_NeuralODE" or alg_type == "FE_NeuralODE_Residuals":
                predictor.compute_representations(example_states, example_actions, example_next_states)

            for episode in range(num_episodes_per_hidden_param):
                # fetch init state for this episode
                init_state = initial_states[episode] # init states are the same for all hidden parameters because we want to compare across hidden parameters
                init_state = convert_quat_state_to_euler(init_state)

                # create the MPCEnv
                seed += 1
                _, info = env.reset(add_sphere=render, seed=seed) # for viz

                # set initial state
                env.set_state(init_state)
                obs = env._get_obs()

                # save initial state
                initial_state = torch.tensor(obs, device=device)
                states_to_save[hidden_param_index, episode, 0] = initial_state.cpu()
                if normalize:
                    initial_state = (initial_state - mean) / std

                mpc_env = MPCEnv(predictor, state_space, action_space, initial_state, example_states, example_actions, example_next_states, mean, std, render_env=env)

                # create the planner
                if mpc_version == "GradCEM":
                    planner = GradCEM(planning_horizon, opt_iters, samples, top_samples, mpc_env, device, action_means)
                elif mpc_version == "CEM":
                    planner = CEM(planning_horizon, opt_iters, samples, top_samples, mpc_env, device, action_means)
                elif mpc_version == "Grad":
                    planner = Grad(planning_horizon, opt_iters, samples, top_samples, mpc_env, device, action_means)
                else:
                    raise ValueError(f"Unknown mpc_version '{mpc_version}'")

                # create a renderer
                if render:
                    out.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))

                # print("True action space:" , env.action_space)
                for time in range(test_horizon):
                    # set the initial state of the env
                    current_state = torch.tensor(obs, device=device)
                    # print(*(f"{t:0.2f}" for t in current_state)) # unnoramlized
                    if normalize:
                        current_state = (current_state - mean) / std
                    planner.env.set_state(current_state)

                    action = planner.forward(batch_size=1).squeeze(0).cpu().numpy()
                    # print(f"\t\t\tAction: {action[0]:0.4f}, {action[1]:0.4f}, {action[2]:0.4f}, {action[3]:0.4f}")
                    # assert (action <= action_space[:, 1].numpy()).all(), f"Action out of bounds {action} {action_space[:, 1].numpy()}"
                    # assert (action >= action_space[:, 0].numpy()).all(), f"Action out of bounds {action} {action_space[:, 0].numpy()}"
                    actions_to_save[hidden_param_index, episode, time] = torch.tensor(action, device="cpu")

                    obs, reward, _, _, info = env.step(action)
                    states_to_save[hidden_param_index, episode, time+1] = torch.tensor(obs, device="cpu")
                    if render:
                        img = env.render()
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out.write(img)

                    # check if z is less than 0
                    if obs[4] < 0:
                        print("Collision")
                        if render:
                            img = cv2.putText(img, "Collision", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                            out.write(img)
                            out.write(img)
                            out.write(img)
                        break
                tbar.update(1)
            env.close()

        if not render:
            torch.save(states_to_save, f"{alg_dir}/mpc_states{'_low_data' if low_data else ''}.pt")
            torch.save(actions_to_save, f"{alg_dir}/mpc_actions{'_low_data' if low_data else ''}.pt")
        if render:
            out.release()
