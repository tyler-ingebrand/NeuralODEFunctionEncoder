import argparse
import os
from typing import Union

import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import trange

# loads the data from the newest directory in logs/ant/policy or logs/cheetah/policy. Or, if load_dir is specified, from there
def load_data(env:str, load_dir:Union[str, None]):
    if not load_dir:
        load_dir = f"logs/{'ant' if env == 'Ant-v3' else 'cheetah'}/policy"
        # list all subdirs, get the newest one
        dirs = [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))]
        latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(load_dir, d)))
        load_dir = os.path.join(load_dir, latest)

    states = torch.load(os.path.join(load_dir, "states.pt"))
    actions = torch.load(os.path.join(load_dir, "actions.pt"))

    # remove contact forces from env
    if env == "Ant-v3":
        states = states[:, :28]

    # organize into trajectories (they are length 1000)
    states = states.view(-1, 1000, states.shape[-1])
    actions = actions.view(-1, 1000, actions.shape[-1])

    return states, actions



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', help='environment ID', default='Ant-v3')
    argparser.add_argument('--seed', help='RNG seed', type=int, default=0)
    argparser.add_argument('--num-timesteps', type=int, default=int(1e6))
    argparser.add_argument('--alg', help='RL algorithm', default='sac')
    argparser.add_argument('--visualize', help='Visualize the policy, does not save data', action='store_true')
    args = argparser.parse_args()
    assert args.env is not None and args.env in ['Ant-v3', 'HalfCheetah-v3']
    assert args.alg is not None and args.alg in ['ppo', 'sac']

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    render_mode = 'human' if args.visualize else None

    # create env
    if args.env == 'Ant-v3':
        from VariableAntEnv import VariableAntEnv
        env = VariableAntEnv({"gravity":(-9.8, -1.0)}, render_mode=render_mode, terminate_when_unhealthy=False)
    else:
        from VariableCheetahEnv import VariableCheetahEnv
        env = VariableCheetahEnv({"torso_length":(0.5, 1.5)}, render_mode=render_mode)

    # load trained policy
    load_dir = f"logs/{'ant' if args.env == 'Ant-v3' else 'cheetah'}/policy"

    # list all subdirs, get the newest one
    dirs = [d for d in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, d))]
    latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(load_dir, d)))
    load_dir = os.path.join(load_dir, latest)


    # load the pretrained model
    model = SAC.load(os.path.join(load_dir, "model"))
    terminated = False

    if args.visualize:
        for timestep in range(args.num_timesteps):
            if terminated or timestep % 1000 == 0:
                obs, info = env.reset()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
    else:
        # create space to store data
        states = torch.zeros((args.num_timesteps, env.observation_space.shape[0]))
        actions = torch.zeros((args.num_timesteps, env.action_space.shape[0]))

        # gather data
        for timestep in trange(args.num_timesteps):
            if timestep % 1000 == 0:
                obs, info = env.reset()
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            states[timestep] = torch.tensor(obs)
            actions[timestep] = torch.tensor(action)

        # save data
        save_dir = load_dir
        torch.save(states, os.path.join(save_dir, "states.pt"))
        torch.save(actions, os.path.join(save_dir, "actions.pt"))

