import os
from datetime import datetime

import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from gather_trajectories import load_data

argparser = argparse.ArgumentParser()
argparser.add_argument("--predictor", type=str, required=True)
argparser.add_argument("--env", type=str, required=True)
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--ignore_actions", action="store_true")
argparser.add_argument("--normalize", action="store_true")
argparser.add_argument("--steps", type=int, default=1000)
argparser.add_argument('--data_type', help='The method to use to gather data', default='random')

args = argparser.parse_args()
assert args.env in ["Ant-v3", "HalfCheetah-v3", "drone"]
assert args.predictor in ["MLP", "NeuralODE", "FE", "FE_NeuralODE", "FE_Residuals", "FE_NeuralODE_Residuals", "Oracle"]
use_actions = not args.ignore_actions
if not use_actions:
    print("Ignoring actions. This will greatly reduce the performance for any data setting except on-policy.")
    print("For on-policy, it will only slightly decrease performance. ")

# seed everything
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load the data
train_states, train_actions, train_next_states, test_states, test_actions, test_next_states, train_hidden_params, test_hidden_params = load_data(args.env, policy_type=args.data_type, normalize=args.normalize, get_groundtruth_hidden_params=True)
state_size, action_size = train_states.shape[-1], train_actions.shape[-1]

if args.predictor == "Oracle": # hidden params only used for oracle
    min_vals_hps = {key: min([x[key] for x in train_hidden_params]) for key in train_hidden_params[0].keys()}
    max_vals_hps = {key: max([x[key] for x in train_hidden_params]) for key in train_hidden_params[0].keys()}
else:
    del train_hidden_params, test_hidden_params


# convert device
all_states_train = train_states
all_actions_train = train_actions
all_next_states_train = train_next_states
all_states_test = test_states
all_actions_test = test_actions
all_next_states_test = test_next_states
del train_states, train_actions, train_next_states, test_states, test_actions, test_next_states

# create a predictor
if args.predictor == "MLP":
    from Predictors.MLP import MLP
    predictor = MLP(state_size, action_size, use_actions=use_actions).to(args.device)
    residuals = False
elif args.predictor == "FE":
    from Predictors.FE import FE
    predictor = FE(state_size, action_size, use_actions=use_actions).to(args.device)
    residuals = False
elif args.predictor == "NeuralODE":
    from Predictors.NeuralODE import NeuralODE
    predictor = NeuralODE(state_size, action_size, use_actions=use_actions).to(args.device)
    residuals = False
elif args.predictor == "FE_NeuralODE":
    from Predictors.FE_NeuralODE import FE_NeuralODE
    predictor = FE_NeuralODE(state_size, action_size, use_actions=use_actions).to(args.device)
    residuals = False
elif args.predictor == "FE_Residuals":
    from Predictors.FE import FE
    predictor = FE(state_size, action_size, use_actions=use_actions).to(args.device)
    residuals = True
elif args.predictor == "FE_NeuralODE_Residuals":
    from Predictors.FE_NeuralODE_Residuals import FE_NeuralODE_Residuals
    predictor = FE_NeuralODE_Residuals(state_size, action_size, use_actions=use_actions).to(args.device)
    residuals = True
elif args.predictor == "Oracle":
    from Predictors.Oracle import Oracle
    predictor = Oracle(state_size, action_size, use_actions=use_actions, min_vals_hps=min_vals_hps, max_vals_hps=max_vals_hps).to(args.device)
    residuals = False
else:
    raise Exception("Unknown predictor")

# optimizer
optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)

# create a logger
datetimestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"logs/{'ant' if args.env == 'Ant-v3' else 'cheetah' if args.env == 'HalfCheetah-v3' else 'drone'}/predictor/{datetimestr}/{args.predictor}/{args.data_type}"
os.makedirs(save_dir, exist_ok=True)
logger = SummaryWriter(save_dir)

# train the predictor
n_envs_at_once = 10
n_example_points = 200
grad_accumulation_steps = 5
for step in trange(args.steps):
    total_loss = 0.0
    for grad_accum_step in range(grad_accumulation_steps):
        # randomize
        # get some envs
        env_indicies = torch.randperm(all_states_train.shape[0])[:n_envs_at_once]

        # get some random steps
        perm = torch.randperm(all_states_train.shape[1] * all_states_train.shape[2])
        example_indicies = perm[:n_example_points]
        train_indicies = perm[n_example_points:][:800] # only gather the first 800 random points

        # convert to episode and timestep indicies
        example_episode_indicies = example_indicies // all_states_train.shape[2]
        example_timestep_indicies = example_indicies % all_states_train.shape[2]
        train_episode_indicies = train_indicies // all_states_train.shape[2]
        train_timestep_indicies = train_indicies % all_states_train.shape[2]

        # gather data
        states = all_states_train[env_indicies, :, :, :][:, train_episode_indicies, train_timestep_indicies, :].to(args.device)
        actions = all_actions_train[env_indicies, :, :, :][:, train_episode_indicies, train_timestep_indicies, :].to(args.device)
        next_states = all_next_states_train[env_indicies, :, :, :][:, train_episode_indicies, train_timestep_indicies, :].to(args.device)
        example_states = all_states_train[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(args.device)
        example_actions = all_actions_train[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(args.device)
        example_next_states = all_next_states_train[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(args.device)


        # train the average predictor, if using residuals method
        if residuals:
            predicted_next_states, resid_info = predictor.predict(states, actions, example_states, example_actions, example_next_states, average_function_only=True)
            loss1 = torch.nn.functional.mse_loss(predicted_next_states, next_states)
            loss1.backward()

        # train the predictor
        if args.predictor == "Oracle":
            predicted_next_states, train_info = predictor.predict(states, actions, hidden_params=[train_hidden_params[i] for i in env_indicies])
        else:
            predicted_next_states, train_info = predictor.predict(states, actions, example_states, example_actions, example_next_states)
        loss2 = torch.nn.functional.mse_loss(predicted_next_states, next_states)
        total_loss += loss2.item() / grad_accumulation_steps
        # if loss2 > 10:
        #     print("?")

        # backprop
        loss2.backward()
        del states, actions, next_states, example_states, example_actions, example_next_states, predicted_next_states

    norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1)
    optimizer.step()
    optimizer.zero_grad()

    if torch.isnan(loss2):
        print("Nan loss")

    # log
    logger.add_scalar("train/loss", total_loss, step)
    logger.add_scalar("train/norm", norm.item(), step)
    for key, val in train_info.items():
        logger.add_scalar(f"train/{key}", val, step)
    if residuals:
        for key, val in resid_info.items():
            logger.add_scalar(f"train_residuals/{key}", val, step)

    # now do the same thing for testing only
    total_test_loss = 0.0
    for grad_accum_step in range(grad_accumulation_steps):
        # randomize
        # get some envs
        env_indicies = torch.randperm(all_states_test.shape[0])[:n_envs_at_once]

        # get some random steps
        perm = torch.randperm(all_states_test.shape[1] * all_states_test.shape[2])
        example_indicies = perm[:n_example_points]
        test_indicies = perm[n_example_points:][:800]  # only gather the first 800 random points

        # convert to episode and timestep indicies
        example_episode_indicies = example_indicies // all_states_test.shape[2]
        example_timestep_indicies = example_indicies % all_states_test.shape[2]
        test_episode_indicies = test_indicies // all_states_test.shape[2]
        test_timestep_indicies = test_indicies % all_states_test.shape[2]

        # gather data
        states = all_states_test[env_indicies, :, :, :][:, test_episode_indicies, test_timestep_indicies,:].to(args.device)
        actions = all_actions_test[env_indicies, :, :, :][:, test_episode_indicies, test_timestep_indicies, :].to(args.device)
        next_states = all_next_states_test[env_indicies, :, :, :][:, test_episode_indicies, test_timestep_indicies, :].to(args.device)
        example_states = all_states_test[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(args.device)
        example_actions = all_actions_test[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(args.device)
        example_next_states = all_next_states_test[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(args.device)

        # reshape to ignore the episode dim, since we are only doing 1 step, it doesnt matter
        states = states.view(states.shape[0], -1, states.shape[-1])
        actions = actions.view(actions.shape[0], -1, actions.shape[-1])
        next_states = next_states.view(next_states.shape[0], -1, next_states.shape[-1])
        example_states = example_states.view(example_states.shape[0], -1, example_states.shape[-1])
        example_actions = example_actions.view(example_actions.shape[0], -1, example_actions.shape[-1])
        example_next_states = example_next_states.view(example_next_states.shape[0], -1, example_next_states.shape[-1])

        # test the predictor
        if args.predictor == "Oracle":
            predicted_next_states, test_info = predictor.predict(states, actions, hidden_params=[test_hidden_params[i] for i in env_indicies])
        else:
            predicted_next_states, test_info = predictor.predict(states, actions, example_states, example_actions, example_next_states)
        test_loss = torch.nn.functional.mse_loss(predicted_next_states, next_states)
        total_test_loss += test_loss.item() / grad_accumulation_steps
        del states, actions, next_states, example_states, example_actions, example_next_states, predicted_next_states, test_loss

    # log
    logger.add_scalar("test/loss", total_test_loss, step)
    for key, val in test_info.items():
        logger.add_scalar(f"test/{key}", val, step)

# save the predictor
torch.save(predictor.state_dict(), os.path.join(save_dir, "model.pt"))