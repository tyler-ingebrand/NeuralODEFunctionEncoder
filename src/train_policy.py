import argparse
import os.path
from datetime import datetime
from multiprocessing import freeze_support

import torch
import numpy as np
import gymnasium as gym
import stable_baselines3
from VariableAntEnv import VariableAntEnv
from VariableCheetahEnv import VariableCheetahEnv
from VariableDroneEnvironment import VariableDroneEnv

if __name__ == '__main__':
    freeze_support()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', help='environment ID', default='Ant-v3')
    argparser.add_argument('--seed', help='RNG seed', type=int, default=0)
    argparser.add_argument('--num-timesteps', type=int, default=int(1e7))
    argparser.add_argument('--alg', help='RL algorithm', default='sac')

    args = argparser.parse_args()
    assert args.env is not None and args.env in ['Ant-v3', 'HalfCheetah-v3', 'drone']
    assert args.alg is not None and args.alg in ['ppo', 'sac']

    # seed everything
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create envs
    if args.env == 'Ant-v3':
        make_env = lambda : VariableAntEnv({}) # defaults
    elif args.env == 'HalfCheetah-v3':
        make_env = lambda : VariableCheetahEnv({}) # defaults
    else:
        make_env = lambda: VariableDroneEnv({})
    vec_env = stable_baselines3.common.env_util.make_vec_env(make_env, n_envs=4)

    # save dir for data and model
    datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    savedir = f"logs/{'ant' if args.env == 'Ant-v3' else 'cheetah' if args.env == 'HalfCheetah-v3' else 'drone'}/policy/{datetime_str}"
    os.makedirs(savedir, exist_ok=True)

    # create model
    if args.alg == 'ppo':
        model = stable_baselines3.PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=savedir)
    elif args.alg == 'sac':
        model = stable_baselines3.SAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=savedir)
    else:
        raise ValueError(f"Unknown algorithm {args.alg}")

    # train and save
    model.learn(total_timesteps=args.num_timesteps, progress_bar=True)
    model.save(os.path.join(savedir, "model"))