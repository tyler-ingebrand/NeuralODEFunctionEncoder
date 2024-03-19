import numpy as np
import torch
from torch import jit
from torch import nn, optim
import os
import cv2
from tqdm import trange
import colorednoise as cn


from gather_trajectories import load_data
from src.CEM_MPC import CEM
from src.Grad_CEM_MPC import GradCEM
from src.MPC_Env import MPCEnv

if __name__ == "__main__":

    # hyper params
    env_str = 'drone'

    # datetime_str = '2024-03-08_21-29-13'
    # alg_type =  'MLP'
    # datetime_str = '2024-03-08_23-09-07'
    # alg_type = 'FE_NeuralODE_Residuals'
    datetime_str = '2024-03-08_21-41-57'
    alg_type = "FE_NeuralODE"

    policy_type = 'random'
    hidden_param_index = 0
    use_actions = True
    test_horizon = 25
    normalize = True
    seed = 0
    use_true_env = False
    device = torch.device("cuda" if torch.cuda.is_available() and not use_true_env else "cpu")

    # mpc params
    planning_horizon = 10
    opt_iters = 100
    samples = 100
    top_samples = int(0.4 * samples)
    mpc_version = "GradCEM"

    # set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # actual code
    # load data
    _, training_actions123, _, states, actions, next_states, train_hidden_params, hidden_params = load_data(env_str, policy_type, normalize=False, get_groundtruth_hidden_params=True)
    _, _, _, _, _, _, mean, std = load_data(env_str, policy_type, normalize=True, get_normalize_params=True)

    # calculate the action limits in the data
    training_actions123 = training_actions123.reshape(-1, training_actions123.shape[-1])
    action_means = training_actions123.mean(dim=0)
    quants = torch.quantile(training_actions123, torch.tensor([0.25, 0.75]), dim=0)
    action_space = torch.stack([quants[0], quants[1]], dim=1)
    del training_actions123

    # create the env
    hidden_params = hidden_params[hidden_param_index]
    hidden_params_dict = {key: (value, value) for key, value in hidden_params.items()}
    if env_str == "HalfCheetah-v3":
        from VariableCheetahEnv import VariableCheetahEnv
        env = VariableCheetahEnv(hidden_params_dict, render_mode="rgb_array")
        true_env = VariableCheetahEnv(hidden_params_dict, render_mode="rgb_array")
    elif env_str == "Ant-v3":
        from VariableAntEnv import VariableAntEnv
        env = VariableAntEnv(hidden_params_dict, render_mode="rgb_array")
        true_env = VariableAntEnv(hidden_params_dict, render_mode="rgb_array")
    elif env_str == "drone":
        from VariableDroneEnvironment import VariableDroneEnv
        env = VariableDroneEnv(hidden_params_dict, render_mode="rgb_array")
        true_env = VariableDroneEnv(hidden_params_dict, render_mode="rgb_array")
    else:
        raise ValueError(f"Unknown env '{env_str}'")
    _ = env.reset()
    _ = true_env.reset()
    img = env.render()
    width, height = img.shape[1], img.shape[0]
    state_space = env.observation_space
    state_shape, action_shape = states.shape[-1], actions.shape[-1]

    # load model
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
        from Predictors.FE_NeuralODE_Fast import FE_NeuralODE_Fast
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

    # efficient models for MPC
    if alg_type == "FE_NeuralODE":
        fast_predictor = FE_NeuralODE_Fast(predictor)
        del predictor
        predictor = fast_predictor

    # get example data points
    num_examples = 200
    perm = torch.randperm(states.shape[1] * states.shape[2])
    example_indicies = perm[:num_examples]
    example_episode_indicies = example_indicies // states.shape[2]
    example_timestep_indicies = example_indicies % states.shape[2]
    example_states = states[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]
    example_actions = actions[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]
    example_next_states = next_states[hidden_param_index, example_episode_indicies, example_timestep_indicies, :]
    if normalize and not use_true_env:
        example_states = (example_states - mean) / std
        example_next_states = (example_next_states - mean) / std

    # convert device of everything
    example_states = example_states.to(device)
    example_actions = example_actions.to(device)
    example_next_states = example_next_states.to(device)
    mean = mean.to(device)
    std = std.to(device)
    predictor = predictor.to(device)

    # update the representation
    if alg_type == "FE_NeuralODE":
        predictor.compute_representations(example_states, example_actions, example_next_states)


    # create the MPCEnv
    obs, info = env.reset()
    initial_state =torch.tensor(obs, device=device)
    if normalize and not use_true_env:
        initial_state = (initial_state - mean) / std

    if not use_true_env:
        system_model = predictor
    else:
        system_model = true_env
    mpc_env = MPCEnv(system_model, state_space, action_space, initial_state, example_states, example_actions, example_next_states, mean, std, render_env=env)

    # create the planner
    if mpc_version == "GradCEM":
        planner = GradCEM(planning_horizon, opt_iters, samples, top_samples, mpc_env, device, action_means)
    elif mpc_version == "CEM":
        planner = CEM(planning_horizon, opt_iters, samples, top_samples, mpc_env, device, action_means)
    else:
        raise ValueError(f"Unknown mpc_version '{mpc_version}'")

    # create a renderer
    width, height, fps = width, height, 10
    out = cv2.VideoWriter(f'{alg_dir}/mpc_trajectory.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    out.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))

    # run the episode
    tbar =  trange(test_horizon)
    terminated = False
    print("True action space:" , env.action_space)
    for time in tbar:
        try:
            # set the initial state of the env
            current_state = torch.tensor(obs, device=device)
            if normalize and not use_true_env:
                current_state = (current_state - mean) / std
            planner.env.set_state(current_state)


            action = planner.forward(batch_size=1).squeeze(0).cpu().numpy()
            # print(action)
            obs, reward, _, _, info = env.step(action)
            img = env.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)

            # check if z is less than 0
            if obs[4] < 0:
                terminated = True
                break
            else:
                terminated = False
            tbar.set_description_str(f"z = {obs[4]:.2f}  ")

        except KeyboardInterrupt:
            break

    if terminated:
        img = cv2.putText(img, "Collision", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        out.write(img)
        out.write(img)
        out.write(img)

    out.release()
