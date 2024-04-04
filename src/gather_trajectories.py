import argparse
import os
from typing import Union

import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange
from ExploratoryPID import ExploratoryPID

# loads the data from the newest directory in logs/ant/policy or logs/cheetah/policy. Or, if load_dir is specified, from there
def load_data(env:str, policy_type:str, load_dir:Union[str, None]=None, normalize=True, get_groundtruth_hidden_params=False, get_normalize_params = False):
    if not load_dir:
        load_dir = f"logs/{'ant' if env == 'Ant-v3' else 'cheetah' if env == 'HalfCheetah-v3' else 'drone'}/policy"
        # list all subdirs, get the newest one
        dirs = [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))]
        latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(load_dir, d)))
        load_dir = os.path.join(load_dir, latest)

        # now add the policy type
        assert policy_type in ["random", "on-policy", "precise", "precise2"]
        load_dir = os.path.join(load_dir, policy_type)

    # load the data
    train_states = torch.load(os.path.join(load_dir, "train_states.pt"))
    train_actions = torch.load(os.path.join(load_dir, "train_actions.pt"))
    train_next_states = torch.load(os.path.join(load_dir, "train_next_states.pt"))
    test_states = torch.load(os.path.join(load_dir, "test_states.pt"))
    test_actions = torch.load(os.path.join(load_dir, "test_actions.pt"))
    test_next_states = torch.load(os.path.join(load_dir, "test_next_states.pt"))

    # if ant, remove contact forces
    if env == "Ant-v3":
        train_states = train_states[:, :, :, :29]
        train_next_states = train_next_states[:, :, :, :29]
        test_states = test_states[:, :, :, :29]
        test_next_states = test_next_states[:, :, :, :29]

    # normalize
    if normalize:
        mean, std = train_states.reshape(-1, train_states.shape[-1]).mean(dim=0), train_states.reshape(-1, train_states.shape[-1]).std(dim=0)
        train_states = (train_states - mean) / std
        test_states = (test_states - mean) / std
        train_next_states = (train_next_states - mean) / std
        test_next_states = (test_next_states - mean) / std

    if get_groundtruth_hidden_params:
        train_hidden_params = torch.load(os.path.join(load_dir, "train_hidden_params.pt"))
        test_hidden_params = torch.load(os.path.join(load_dir, "test_hidden_params.pt"))
        return train_states, train_actions, train_next_states, test_states, test_actions, test_next_states, train_hidden_params, test_hidden_params
    elif get_normalize_params:
        return train_states, train_actions, train_next_states, test_states, test_actions, test_next_states, mean, std
    else:
        return train_states, train_actions, train_next_states, test_states, test_actions, test_next_states


def visualize(train_env, args):
    terminated = False
    for timestep in range(10000):
        if terminated or timestep % 1000 == 0:
            obs, info = train_env.reset()
        action = train_env.action_space.sample()
        nobs, reward, terminated, truncated, info = train_env.step(action)

def gather_data_with_hidden_params(env, model, n_envs, episode_length, n_episodes, data_type):
    # create space to store data
    states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
    actions = torch.zeros((n_envs, n_episodes, episode_length, env.action_space.shape[0]))
    next_states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
    hidden_params = []
    transitions_per_env = n_episodes * episode_length

    # gather data
    for env_index in trange(n_envs):
        obs, info = env.reset()  # resets hidden params to something new
        hidden_params.append(info["dynamics"])
        for step_index in range(transitions_per_env):
            # reset every so many steps
            if step_index % episode_length == 0:
                obs, info = env.reset(reset_hps=False)

            # get action
            if data_type == "random":
                action = env.action_space.sample()
            else:  # on policy
                action, _states = model.predict(obs, deterministic=True)

            # transition
            nobs, reward, terminated, truncated, info = env.step(action)

            # store data
            states[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(obs)
            actions[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(action)
            next_states[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(nobs)

            # go next
            obs = nobs
    assert len(hidden_params) == n_envs
    return states, actions, next_states, hidden_params

# note the inputs (states, actions) do not depend on any hidden parameters
# they are sampled from the default environment
# but the transition itself (next state) does depend on the hidden parameters
def gather_data_without_hidden_params(env, model, n_envs, episode_length, n_episodes, data_type):

    # create space to store data
    states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
    actions = torch.zeros((n_envs, n_episodes, episode_length, env.action_space.shape[0]))
    next_states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
    hidden_params = []
    transitions_per_env = n_episodes * episode_length

    if data_type == "precise":
        # gather a set of states actions using random policy
        state, info = env.reset()
        mujoco_state_shape = env.env.sim.get_state().flatten().shape

        # mujoco states
        mujoco_states = torch.zeros((transitions_per_env, mujoco_state_shape[0]))
        mujoco_actions = torch.zeros((transitions_per_env, env.action_space.shape[0]))

        for step_index in trange(transitions_per_env):
            if step_index % episode_length == 0:
                obs, info = env.reset()

            # get action
            action = env.action_space.sample()
            mujoco_state = env.env.sim.get_state().flatten()
            mujoco_states[step_index] = torch.tensor(mujoco_state)
            mujoco_actions[step_index] = torch.tensor(action)

            # step env
            obs, reward, terminated, truncated, info = env.step(action)

        # then do this transition for every hidden parameter, so the initial conditions are exactly the same
        for env_index in trange(n_envs):
            _, info = env.reset()  # samples a hidden param
            hidden_params.append(info["dynamics"])

            for step_index in range(transitions_per_env):
                # get initial conditions
                env.env.sim.set_state_from_flattened(mujoco_states[step_index].numpy())
                action = mujoco_actions[step_index].numpy()

                # get observation and next observation
                obs = env.env.unwrapped._get_obs()
                nobs, _, _, _, _ = env.step(action)

                # store data
                states[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(obs)
                actions[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(action)
                next_states[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(nobs)

    elif data_type == "precise2":
        # gather a set of states actions using random policy
        state, info = env.reset()
        mujoco_state_shape = env.env.sim.get_state().flatten().shape

        # mujoco states
        mujoco_states = torch.zeros((transitions_per_env * n_envs, mujoco_state_shape[0]))
        mujoco_actions = torch.zeros((transitions_per_env * n_envs, env.action_space.shape[0]))

        for step_index in trange(transitions_per_env * n_envs):
            if step_index % 1000 == 0:
                obs, info = env.reset()

            # get action
            action = env.action_space.sample()
            mujoco_state = env.env.sim.get_state().flatten()
            mujoco_states[step_index] = torch.tensor(mujoco_state)
            mujoco_actions[step_index] = torch.tensor(action)

            # step env
            obs, reward, terminated, truncated, info = env.step(action)

        # then do this transition for every hidden parameter, so the initial conditions are exactly the same
        for env_index in trange(n_envs):
            _, info = env.reset()
            hidden_params.append(info["dynamics"])
            for step_index in range(transitions_per_env):
                # get initial conditions
                mujoco_state = mujoco_states[env_index * transitions_per_env + step_index]
                env.env.sim.set_state_from_flattened(mujoco_state.numpy())
                action = mujoco_actions[env_index * transitions_per_env + step_index].numpy()

                # get observation and next observation
                obs = env.env.unwrapped._get_obs()
                nobs, _, _, _, _ = env.step(action)

                # store data
                states[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(obs)
                actions[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(action)
                next_states[env_index, step_index // episode_length, step_index % episode_length] = torch.tensor(nobs)
    else:
        raise Exception("Invalid data type")
    assert len(hidden_params) == n_envs
    return states, actions, next_states, hidden_params

def drone_pid_data_gathering(env, n_envs, episode_length, n_episodes):
    with torch.no_grad():
        # import os, psutil
        # process = psutil.Process()
        # create space to store data
        states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
        actions = torch.zeros((n_envs, n_episodes, episode_length, env.action_space.shape[0]))
        next_states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
        hidden_params = []
        transitions_per_env = n_episodes * episode_length

        # import psutil

        # gather data
        tbar = trange(n_envs)
        for env_index in tbar:
            # print(process.memory_info().rss / 1e6 ,"MB")  # in bytes
            obs, info = env.reset()  # resets hidden params to something new
            hps = info["dynamics"]
            hps = {key: (value, value) for key, value in hps.items()}

            episode_index = 0
            while episode_index < n_episodes:
                tbar.set_description(f"Episode {episode_index}")
                # print(psutil.Process().memory_info().rss / 1e6, " MB")
                # soft reset env to get a new goal location
                obs, info = env.reset(reset_hps=False)

                # create a new pid
                ctrl_env_create = lambda : VariableDroneEnv(hps, render_mode=None)
                ctrl = ExploratoryPID(env_func=ctrl_env_create)

                # create buffer for this episodes data
                episode_states = torch.zeros((episode_length, env.observation_space.shape[0]))
                episode_actions = torch.zeros((episode_length, env.action_space.shape[0]))
                episode_next_states = torch.zeros((episode_length, env.observation_space.shape[0]))

                # make sure it did not terminate early
                terminated_early = False

                # run the pid, store results
                for step_index in range(episode_length):
                    # get action
                    action = ctrl.select_action(info["euler_obs"], info)
                    # noise = np.random.normal(-0.002, 0.002, action.shape)
                    # action += noise
                    # action = np.clip(action, env.action_space.low, env.action_space.high)
                    # the pid controller handles noise

                    # transition
                    nobs, reward, terminated, truncated, info = env.step(action)

                    if step_index < episode_length - 1 and (terminated or truncated):
                        terminated_early = True
                        break

                    # store data
                    episode_states[step_index] = torch.tensor(obs)
                    episode_actions[step_index] = torch.tensor(action)
                    episode_next_states[step_index] = torch.tensor(nobs)

                    # go next
                    obs = nobs

                # make sure to free memory, bullet does not do it for you
                ctrl.env.close()
                del ctrl, ctrl_env_create

                # check if episode terminated early
                if terminated_early:
                    continue
                else:
                    # save data and hps
                    states[env_index, episode_index] = episode_states
                    actions[env_index, episode_index] = episode_actions
                    next_states[env_index, episode_index] = episode_next_states
                    episode_index += 1
            hidden_params.append(info["dynamics"])


        assert len(hidden_params) == n_envs
        return states, actions, next_states, hidden_params

def drone_thorough_data_gathering(env, n_envs, episode_length, n_episodes):
    with torch.no_grad():
        states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
        actions = torch.zeros((n_envs, n_episodes, episode_length, env.action_space.shape[0]))
        next_states = torch.zeros((n_envs, n_episodes, episode_length, env.observation_space.shape[0]))
        hidden_params = []

        # x, dx, y, dy, z, dz, phi, theta, psi, dphi, dtheta, dpsi
        obs_high = np.array([2, 5,   2, 5,   2, 1.05,       1.5, 1.5, 3.14, 30, 30, 30]) # note these use euler angles
        obs_low = np.array([-2, -5, -2, -5,  0.05, -3.5,    -1.5, -1.5, -3.14, -30, -30, -30])

        # gather data
        tbar = trange(n_envs)
        for env_index in tbar:
            # print(process.memory_info().rss / 1e6 ,"MB")  # in bytes
            obs, info = env.reset()  # resets hidden params to something new

            for episode_index in range(n_episodes):
                tbar.set_description(f"Episode {episode_index}")
                for time_index in range(episode_length):
                    # get a random state
                    obs = np.random.rand(obs_high.shape[0]) * (obs_high - obs_low) + obs_low
                    env.set_state(obs)
                    obs = env._get_obs()

                    # get random  action
                    action = env.action_space.sample()

                    # transition
                    nobs, reward, terminated, truncated, info = env.step(action)

                    # store data
                    states[env_index, episode_index, time_index] = torch.tensor(obs)
                    actions[env_index, episode_index, time_index] = torch.tensor(action)
                    next_states[env_index, episode_index, time_index] = torch.tensor(nobs)

            hidden_params.append(info["dynamics"])
        assert len(hidden_params) == n_envs
        return states, actions, next_states, hidden_params



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', help='environment ID', default='Ant-v3')
    argparser.add_argument('--seed', help='RNG seed', type=int, default=0)
    argparser.add_argument('--num_envs', type=int, default=int(1000))
    argparser.add_argument('--transitions_per_env', type=int, default=int(10_000))
    argparser.add_argument('--alg', help='RL algorithm', default='sac')
    argparser.add_argument('--data_type', help='The method to use to gather data', default='random')
    argparser.add_argument('--visualize', help='Visualize the policy, does not save data', action='store_true')
    argparser.add_argument('--xdim', help='To include x dim', action='store_false')
    args = argparser.parse_args()
    assert args.env is not None and args.env in ['Ant-v3', 'HalfCheetah-v3', 'drone']
    assert args.alg is not None and args.alg in ['ppo', 'sac']
    assert args.data_type is not None and args.data_type in ['random', 'on-policy', "precise", "precise2"]
    use_x_dim = args.xdim

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    render_mode = 'human' if args.visualize else None

    # create env
    if args.env == 'Ant-v3':
        from VariableAntEnv import VariableAntEnv
        ANKLE_LENGTH = (2*0.4**2)**0.5
        LEG_LENGTH = (2*0.2**2)**0.5
        gear_max, gear_min = 60, 60
        foot_min, foot_max  = ANKLE_LENGTH / 2, ANKLE_LENGTH * 1.5  # +/- 50%
        leg_min, leg_max = LEG_LENGTH, LEG_LENGTH * 1.5
        vars = {    "gravity": (-9.8,  -9.8),
                    "front_left_leg_length": (leg_min, leg_max),
                    "front_left_foot_length": (foot_min, foot_max),
                    "front_right_leg_length": (leg_min, leg_max),
                    "front_right_foot_length": (foot_min, foot_max),
                    "back_left_leg_length": (leg_min, leg_max),
                    "back_left_foot_length": (foot_min, foot_max),
                    "back_right_leg_length": (leg_min, leg_max),
                    "back_right_foot_length": (foot_min, foot_max),
                    "front_left_gear": (gear_min, gear_max),
                    "front_right_gear": (gear_min, gear_max),
                    "back_left_gear": (gear_min, gear_max),
                    "back_right_gear": (gear_min, gear_max),
                    "front_left_ankle_gear": (gear_min, gear_max),
                    "front_right_ankle_gear": (gear_min, gear_max),
                    "back_left_ankle_gear": (gear_min, gear_max),
                    "back_right_ankle_gear": (gear_min, gear_max),
            }
        train_env = VariableAntEnv(vars, render_mode=render_mode, terminate_when_unhealthy=False, exclude_current_positions_from_observation=not use_x_dim)
        test_env1 = VariableAntEnv(vars, render_mode=render_mode, terminate_when_unhealthy=False, exclude_current_positions_from_observation=not use_x_dim)
        test_env2 = VariableAntEnv(vars, render_mode=render_mode, terminate_when_unhealthy=False, exclude_current_positions_from_observation=not use_x_dim)

    elif args.env == 'HalfCheetah-v3':
        from VariableCheetahEnv import VariableCheetahEnv

        vars = {
            'friction': (0.0, 1.0),
            'torso_length': (1 * 0.5, 1 * 1.5),
            'bthigh_length': (0.145 * 0.5, 0.145 * 1.5),
            'bshin_length': (0.15 * 0.5, 0.15 * 1.5),
            'bfoot_length': (0.094 * 0.5, 0.094 * 1.5),
            'fthigh_length': (0.133 * 0.5, 0.133 * 1.5),
            'fshin_length': (0.106 * 0.5, 0.106 * 1.5),
            'ffoot_length': (0.07 * 0.5, 0.07 * 1.5),
            'bthigh_gear': (0, 120 * 1.5),
            'bshin_gear': (0, 90 * 1.5),
            'bfoot_gear': (0, 60 * 1.5),
            'fthigh_gear': (0, 120 * 1.5),
            'fshin_gear': (0, 60 * 1.5),
            'ffoot_gear': (0, 30 * 1.5),
        }
        train_env = VariableCheetahEnv(vars, render_mode=render_mode, exclude_current_positions_from_observation=not use_x_dim)
        test_env1 = VariableCheetahEnv(vars, render_mode=render_mode, exclude_current_positions_from_observation=not use_x_dim)
        test_env2 = VariableCheetahEnv(vars, render_mode=render_mode, exclude_current_positions_from_observation=not use_x_dim)
    else:
        from VariableDroneEnvironment import VariableDroneEnv
        # vars = {
        #     'M': (0.022, 0.032),
        #     "Ixx": (1.3e-5, 1.5e-5),
        #     "Iyy": (1.3e-5, 1.5e-5),
        #     "Izz": (2.1e-5, 2.2e-5)
        #    }
        vars = {'M': (0.02, 0.032),  # .032),
               "Ixx": (1.4e-5, 1.4e-5),
               "Iyy": (1.4e-5, 1.4e-5),
               "Izz": (2.15e-5, 2.15e-5),
               }
        train_env = VariableDroneEnv(vars, render_mode=render_mode)
        test_env1 = VariableDroneEnv(vars, render_mode=render_mode, no_crash=True)
        test_env2 = VariableDroneEnv(vars, render_mode=render_mode, no_crash=True)

    # load trained policy
    load_dir = f"logs/{'ant' if args.env == 'Ant-v3' else 'cheetah' if args.env == 'HalfCheetah-v3' else 'drone'}/policy"

    # list all subdirs, get the newest one
    dirs = [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))]
    latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(load_dir, d)))
    load_dir = os.path.join(load_dir, latest)

    # load the pretrained model
    model = SAC.load(os.path.join(load_dir, "model"))
    terminated = False

    # get parameters
    n_envs = args.num_envs
    episode_length = 125 if args.env == 'drone' else 1000
    n_episodes = args.transitions_per_env // episode_length
    data_type = args.data_type
    test_envs = max(1, n_envs // 10)

    if args.visualize:
        print("This is only to see whats happening, does not generate data")
        visualize(train_env, args)
        exit()
    if args.env == "drone":
        test_states1, test_actions1, test_next_states1, test_hidden_params1 = drone_pid_data_gathering(test_env1,test_envs,episode_length,n_episodes)
        test_states2, test_actions2, test_next_states2, test_hidden_params2 = drone_pid_data_gathering(test_env2,test_envs,episode_length,n_episodes)
        train_states, train_actions, train_next_states, train_hidden_params = drone_pid_data_gathering(train_env, n_envs, episode_length, n_episodes)
    elif args.data_type == "random" or args.data_type == "on-policy":
        train_states, train_actions, train_next_states, train_hidden_params = gather_data_with_hidden_params(train_env, model, n_envs, episode_length, n_episodes, data_type)
        test_states1, test_actions1, test_next_states1, test_hidden_params1 = gather_data_with_hidden_params(test_env1, model, test_envs, episode_length, n_episodes, data_type)
        test_states2, test_actions2, test_next_states2, test_hidden_params2 = gather_data_with_hidden_params(test_env2, model, test_envs, episode_length, n_episodes, data_type)
    else:
        train_states, train_actions, train_next_states, train_hidden_params = gather_data_without_hidden_params(train_env, model, n_envs, episode_length, n_episodes, data_type)
        test_states1, test_actions1, test_next_states1, test_hidden_params1 = gather_data_without_hidden_params(test_env1, model, test_envs, episode_length, n_episodes, data_type)
        test_states2, test_actions2, test_next_states2, test_hidden_params2 = gather_data_without_hidden_params(test_env2, model, test_envs, episode_length, n_episodes, data_type)

    # join testing data
    test_states = torch.cat([test_states1, test_states2], dim=0)
    test_actions = torch.cat([test_actions1, test_actions2], dim=0)
    test_next_states = torch.cat([test_next_states1, test_next_states2], dim=0)

    # save data
    save_dir = os.path.join(load_dir, args.data_type)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(train_states, os.path.join(save_dir, "train_states.pt"))
    torch.save(train_actions, os.path.join(save_dir, "train_actions.pt"))
    torch.save(train_next_states, os.path.join(save_dir, "train_next_states.pt"))
    torch.save(train_hidden_params, os.path.join(save_dir, "train_hidden_params.pt"))

    torch.save(test_states, os.path.join(save_dir, "test_states.pt"))
    torch.save(test_actions, os.path.join(save_dir, "test_actions.pt"))
    torch.save(test_next_states, os.path.join(save_dir, "test_next_states.pt"))
    torch.save(test_hidden_params1 + test_hidden_params2, os.path.join(save_dir, "test_hidden_params.pt"))