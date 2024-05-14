import cv2
import torch
import os

from tqdm import trange

from gather_trajectories import load_data
import matplotlib.pyplot as plt

from src.VariableDroneEnvironment import VariableDroneEnv


def get_returns(states, actions):
    # remove initial state
    states = states[:, :, 1:, :]

    # if drone crashed, some states are all 0s
    # make these equal to the state before, since this should be penalized
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            for k in range(states.shape[2]):
                if (states[i, j, k, :] == torch.zeros(13)).all():
                    states[i, j, k, :] = states[i, j, k - 1, :]


    
    # hover action
    hover_action = 0.07148927728325129  # this is emprically tested to be close to hover for a average weight env
    hover_actions = torch.tensor([hover_action, hover_action, hover_action, hover_action], device=states.device)
    distance_from_hover = torch.clamp((actions - hover_actions).abs() - 0.012, min=0)
    close_to_hover_reward = -torch.norm(distance_from_hover, dim=3)
    
    with torch.no_grad():
        value_vector = torch.ones(states.shape[2], device=states.device)
        value_vector[-1] = 10.0

    # distance from last state to 0,0,1
    x = states[:, :, :, 0]
    y = states[:, :, :, 2]
    z = states[:, :, :, 4]
    distance = torch.sqrt(x ** 2 + y ** 2 + (z - 1) ** 2)
    distance = distance * value_vector
    reward_distance = -distance

    # low velocity
    x_dot = states[:, :, :, 1]
    y_dot = states[:, :, :, 3]
    z_dot = states[:, :, :, 5]
    x_dot = torch.clamp(x_dot.abs() - 0.0, min=0.0) # gives a tolerance of 0.1 velocity without penalty
    y_dot = torch.clamp(y_dot.abs() - 0.0, min=0.0)
    z_dot = torch.clamp(z_dot.abs() - 0.0, min=0.0)
    velocity = x_dot ** 2 + y_dot ** 2 + z_dot ** 2
    velocity = velocity * value_vector
    reward_velocity = -velocity

    # keep it stable every step so it doesnt turb over
    quat1 = states[:, :, :, 6]
    quat2 = states[:, :, :, 7]
    quat3 = states[:, :, :, 8]
    quat4 = states[:, :, :, 9]

    # normalize quats
    quat_norm = torch.sqrt(quat1 ** 2 + quat2 ** 2 + quat3 ** 2 + quat4 ** 2)
    quat1 = quat1 / quat_norm
    quat2 = quat2 / quat_norm
    quat3 = quat3 / quat_norm
    quat4 = quat4 / quat_norm


    # psi = torch.atan2(2 * (quat1 * quat2 + quat3 * quat4), 1 - 2 * (quat2 ** 2 + quat3 ** 2))
    theta = -torch.asin(2 * (quat1 * quat3 - quat4 * quat2))
    phi = torch.atan2(2 * (quat1 * quat4 + quat2 * quat3), 1 - 2 * (quat3 ** 2 + quat4 ** 2))
    phi = torch.where(phi > torch.pi / 2, phi - torch.pi, phi)
    phi = torch.where(phi < -torch.pi / 2, phi + torch.pi, phi)
    phi = phi * -1

    if torch.isnan(phi).any():
        print("phi has nan")
        nan_indices = torch.isnan(phi)
        # get the states and actions that caused the nan
        print(states[nan_indices])
        raise Exception("Nan in phi")
    elif torch.isnan(theta).any():
        print("theta has nan")
        nan_indices = torch.isnan(theta)
        # get the states and actions that caused the nan
        print(states[nan_indices])
        raise Exception("Nan in theta")
    
    # check actions should cancel out
    a1 = actions[:, :, :, 0]
    a2 = actions[:, :, :, 1]
    a3 = actions[:, :, :, 2]
    a4 = actions[:, :, :, 3]
    a_mid1 = (a1 + a3) / 2
    a_mid2 = (a2 + a4) / 2
    assert (a1 - a_mid1 + a3 - a_mid1).all() < 1e-5, "Actions do not cancel out"
    assert (a2 - a_mid2 + a4 - a_mid2).all() < 1e-5, "Actions do not cancel out"

    stability = phi ** 2 + theta ** 2  # + psi ** 2
    stability = stability * value_vector
    reward_stability = -stability

    # penalizes bang bang actions
    dif_actions_1and3 = (a1 - a3)**2
    dif_actions_2and4 = (a2 - a4)**2
    dif_actions = dif_actions_1and3 + dif_actions_2and4
    dif_actions_reward = -dif_actions

    # penalize slew rate
    # action_change = actions[:, :, 1:] - actions[:, :, :-1]
    # action_change = action_change ** 2
    # action_change = -action_change.mean(dim=3)
    # reward_action_change = torch.concat([torch.zeros(action_change.shape[0], action_change.shape[1], 1, device=action_change.device), action_change], dim=2)

    # tune this
    weight_distance = 3
    weight_stability = 5
    weight_velocity = 1
    weight_hover = 0 # 100
    weight_action_change = 2000 # needed to prevent instability at goal locatin
    weight_slew_rate = 1000


    if torch.isnan(reward_stability).any():
        print("reward_distance has nan")
        nan_indices = torch.isnan(reward_stability)
        # get the states and actions that caused the nan
        print(states[nan_indices])
        err

    assert not torch.isnan(reward_distance).any(), "reward_distance has nan"
    assert not torch.isnan(reward_stability).any(), "reward_stability has nan"
    assert not torch.isnan(reward_velocity).any(), "reward_velocity has nan"
    assert not torch.isnan(close_to_hover_reward).any(), "close_to_hover_reward has nan"
    scales = [weight_distance * reward_distance,
                weight_stability * reward_stability,
                weight_velocity * reward_velocity,
                weight_hover * close_to_hover_reward,
                weight_action_change * dif_actions_reward,
                # weight_slew_rate * reward_action_change,


                ]
    traj_return = torch.zeros_like(scales[0])
    for scale in scales:
        traj_return += scale
    return traj_return



env_str = 'drone'
load_dir = "logs/drone/predictor"
policy_type = 'random'
low_data = False


_, _, _, _, _, _, _, test_hidden_params = load_data(env_str, policy_type, normalize=False, get_groundtruth_hidden_params=True)

# fetch all the subdirectories
alg_dirs = [os.path.join(load_dir, x) for x in os.listdir(load_dir) if os.path.isdir(os.path.join(load_dir, x))]

results = {}
for alg_dir in alg_dirs:
    # load the subdir. There should be only 1
    subdirs = [x for x in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, x))]
    assert len(subdirs) == 1, f"Expected 1 subdir, got {len(subdirs)}"
    alg_type = subdirs[0]
    alg_dir = os.path.join(alg_dir, alg_type, policy_type)
    

    # try to load the results
    try:
        extension = '_low_data' if alg_type != 'NeuralODE' and low_data else ''
        states = torch.load(os.path.join(alg_dir, f'mpc_states{extension}.pt'))
        actions = torch.load(os.path.join(alg_dir, f'mpc_actions{extension}.pt'))
    except:
        print(f"Could not load results for {alg_dir}")
        continue

    # skip
    if states.shape != (40, 5, 101, 13): # these are tempory test runs, skip them
        continue

    # get results
    returns = get_returns(states, actions)
    results[alg_type] = returns

# now plot
fig, ax = plt.subplots(figsize=(10, 10))
for alg_type, returns in results.items():
    
    # sum over all timesteps
    returns = returns [:, :, :100]
    returns = returns.sum(dim=2)
    print("Alg type", alg_type, "Mean return", returns.mean().item(), "Min return", returns.min().item(), "Max return", returns.max().item())

    # compute quantiles to account for crashes
    quarts = torch.quantile(returns, torch.tensor([0.0, 0.5, 1.0]), dim=1).transpose(0, 1)
    means = returns.mean(dim=1)
    stds = returns.std(dim=1)
    quarts_with_hps = [(x, test_hidden_params[i]) for i, x in enumerate(quarts)]
    means_with_hps = [(x, test_hidden_params[i]) for i, x in enumerate(means)]
    stds_with_hps = [(x, test_hidden_params[i]) for i, x in enumerate(stds)]

    # sort based on hps for graph
    quarts_with_hps.sort(key=lambda x: x[1]['M'])
    means_with_hps.sort(key=lambda x: x[1]['M'])
    stds_with_hps.sort(key=lambda x: x[1]['M'])


    # plot min, max, median
    xs = [x[1]['M'] for x in quarts_with_hps]# [::4]
    mins = [x[0][0] for x in quarts_with_hps]# [::4]
    medians = [x[0][1] for x in quarts_with_hps]# [::4]
    maxs = [x[0][2] for x in quarts_with_hps]# [::4]
    means = [x[0] for x in means_with_hps]# [::4]
    stds = [x[0] for x in stds_with_hps]# [::4]
    
    # plot
    ax.plot(xs, medians, label=alg_type)
    ax.fill_between(xs, mins, maxs, alpha=0.2)

    # ax.plot(xs, means, label=alg_type)
    # ax.fill_between(xs, [m-s for m,s in zip(means, stds)], [m+s for m,s in zip(means, stds)], alpha=0.2)

ax.set_ylim(-100, 0)
ax.legend(loc="lower left")
ax.set_xlabel('M')
ax.set_ylabel('Average Return')
plt.savefig(f'{load_dir}/mpc_results.png')


# err
# render one of the trajectories
# dir = "logs/drone/predictor/2024-04-01_14-45-23"
# alg_type = "FE_NeuralODE_Residuals"
dir = "logs/drone/predictor/2024-04-01_07-36-10"
alg_type = "FE_NeuralODE"
# dir = "logs/drone/predictor/2024-04-01_07-06-35"
# alg_type = "NeuralODE"
print("Rendering episode ", alg_type)

# load states
alg_dir = os.path.join(dir, alg_type, "random")
states = torch.load(os.path.join(alg_dir, 'mpc_states.pt'))
actions = torch.load(os.path.join(alg_dir, 'mpc_actions.pt'))

# compute returns
returns = get_returns(states, actions)
returns = returns.sum(dim=2)

# get 0th and 1st index of the worst return
worst_index = returns.argmin()
unraveled = torch.unravel_index(worst_index, returns.shape)
hp_index = unraveled[0].item()
traj_index = unraveled[1].item()
print(f"Worst return: {returns[hp_index, traj_index].item()} with Mass={test_hidden_params[hp_index]['M']}")
hp_index = 20
print("HP index", hp_index, "Traj index", traj_index)

# create render env
env = VariableDroneEnv({}, render_mode="human", seed=0)
env.reset(add_sphere=True)
img = env.render()
width, height = img.shape[1] * 2, img.shape[0] * 2

# create renderer
out = cv2.VideoWriter('failed_episode.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
traj_index = None
for t in trange(states.shape[2]):
    imgs = []
    for traj_index in range(1, 5):
        # get state
        s = states[hp_index, traj_index, t, :]
        x, dx, y, dy, z, dz = s[0], s[1], s[2], s[3], s[4], s[5]
        q1, q2, q3, q4 = s[6], s[7], s[8], s[9]
        dphi, dtheta, dpsi = s[10], s[11], s[12]

        # get euler
        q_norm = torch.sqrt(q1 ** 2 + q2 ** 2 + q3 ** 2 + q4 ** 2)
        q1 = q1 / q_norm
        q2 = q2 / q_norm
        q3 = q3 / q_norm
        q4 = q4 / q_norm
        phi = torch.atan2(2 * (q1 * q4 + q2 * q3), 1 - 2 * (q3 ** 2 + q4 ** 2))
        theta = -torch.asin(2 * (q1 * q3 - q4 * q2))
        psi = torch.atan2(2 * (q1 * q2 + q3 * q4), 1 - 2 * (q2 ** 2 + q3 ** 2))

        # fix phi
        phi = torch.where(phi > torch.pi / 2, phi - torch.pi, phi)
        phi = torch.where(phi < -torch.pi / 2, phi + torch.pi, phi)
        phi = phi * -1

        # set state
        env.set_state([ a.item() for a in [x, dx, y, dy, z, dz, phi, theta, psi, dphi, dtheta, dpsi]])
        img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        # write actions to image
        if t < actions.shape[2]:
            action = actions[hp_index, traj_index, t, :]
            txt = f"{action[0].item():0.2f}, {action[1].item():0.2f}, {action[2].item():0.2f}, {action[3].item():0.2f}"
            img = cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        imgs.append(img)
    # make 2x2 pane
    img1 = cv2.hconcat(imgs[:2])
    img2 = cv2.hconcat(imgs[2:])
    img = cv2.vconcat([img1, img2])
    out.write(img)

out.release()