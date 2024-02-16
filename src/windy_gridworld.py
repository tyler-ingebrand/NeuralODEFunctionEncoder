import os
from itertools import chain

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from datetime import datetime
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("--train_type", type=str, default="uniform")
argparse.add_argument("--test_type", type=str, default="uniform")
argparse.add_argument("--use_approximate_model", action="store_false")
argparse.add_argument("--grad_steps", type=int, default=1000)
argparse.add_argument("--plot_time", type=int, default=500)
args = argparse.parse_args()

# seed everything
torch.manual_seed(0)
np.random.seed(0)

# hyper params
wind_range = [-0.2, 0.2]
x_range = [-2.0, 2.0]
y_range = [-2.0, 2.0]
t_range = [0, 10]
num_functions = 10

# nn params
input_size = 4
hidden_size = 100
output_size = 4
device = "cuda:0"
embed_size = 11
descent_steps = args.grad_steps
plot_time = args.plot_time
use_approximate_model = args.use_approximate_model
train_type = args.train_type
test_type = args.test_type

assert train_type in ["uniform", "trajectory", "trajectory_ls"]
assert test_type in ["uniform", "trajectory", "trajectory_ls"]
date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = f"logs/windy_gridworld/{date_time_str}/{train_type}_{test_type}{'_approx' if use_approximate_model else ''}"
os.makedirs(logdir, exist_ok=True)
logger = SummaryWriter(logdir)


# we use rk4
def integrate(model, x_0, t_dif, absolute=True):
    """
    Integrate the model from t_0 to t_f with initial condition x_0
    """
    if len(t_dif.shape) == 1:
        t_dif = t_dif.reshape(1, t_dif.shape[0], 1)
    k1 = model(x_0)
    k2 = model(x_0 + 0.5 * t_dif * k1)
    k3 = model(x_0 + 0.5 * t_dif * k2)
    k4 = model(x_0 + t_dif * k3)
    if absolute:
        return x_0 + t_dif / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    else: # returns the distance between states
        check = t_dif / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if torch.isnan(check).any():
            print("nan")
        return t_dif / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def get_windy_models(number_models, wind_range, absolute=True):
    winds = torch.rand(number_models, 2, device=device) * (wind_range[1] - wind_range[0]) + wind_range[0]
    def windy_gridworld(x):
        if len(x.shape) == 2:
            d_x_pos = x[:, 2]
            d_y_pos = x[:, 3]
            d_x_vel = winds[:, 0]
            d_y_vel = winds[:, 1]
            return torch.concat([d_x_pos.unsqueeze(1), d_y_pos.unsqueeze(1), d_x_vel.unsqueeze(1), d_y_vel.unsqueeze(1)], dim=1)
        else: # 3 dims
            d_x_pos = x[:, :, 2]
            d_y_pos = x[:, :, 3]
            d_x_vel = winds[:, 0].repeat(x.shape[1], 1).transpose(0, 1)
            d_y_vel = winds[:, 1].repeat(x.shape[1], 1).transpose(0, 1)
            if d_x_vel.shape[0] != x.shape[0]:
                d_x_vel = d_x_vel.repeat(x.shape[0], 1)
                d_y_vel = d_y_vel.repeat(x.shape[0], 1)
            return torch.concat((d_x_pos.unsqueeze(-1), d_y_pos.unsqueeze(-1), d_x_vel.unsqueeze(-1), d_y_vel.unsqueeze(-1)), dim=2)

    return lambda x_0, t_dif: integrate(windy_gridworld, x_0, t_dif, absolute=absolute), winds

# the approximate model if it is used
approximate_model = get_windy_models(1, [0,0], absolute=False)[0]

def gather_trajectories(model, winds, x_0, t_length=10.0, num_points=1000):
    # random timepoints between 0 and length
    timepoints = torch.sort(torch.rand(num_points) * t_length).values
    ts = torch.cat([torch.tensor([0.0]), timepoints]).to(device)

    # initialize space for states
    xs = torch.zeros(winds.shape[0], num_points + 1, 4, device=device)
    xs[:, 0, :] = x_0

    # gather data
    for i in range(num_points):
        t_dif = ts[i+1] - ts[i]
        x_now = xs[:, i, :]
        x_next = model(x_now, t_dif)
        xs[:, i+1, :] = x_next
        if torch.isnan(x_next).any():
            print("nan")
    return xs, ts

def get_encodings(basis_functions, vanderpol_models, winds, type="uniform"):
    if type == "uniform":
        return get_encoding_uniform(basis_functions, vanderpol_models, winds)
    elif type == "trajectory":
        return get_encodings_trajectory(basis_functions, vanderpol_models, winds)
    elif type == "trajectory_ls":
        return get_encodings_trajectory_ls(basis_functions, vanderpol_models, winds)
    else:
        raise ValueError(f"Invalid type '{type}'")


def get_encoding_uniform(basis_functions, wind_models, winds):
    xs = torch.rand(winds.shape[0], 1000, output_size, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    t_dif = torch.tensor(0.1).to(device)
    xs_next = wind_models(xs, t_dif)


    true_x_difs = (xs_next - xs)
    if use_approximate_model:
        approximate_difs = approximate_model(xs, t_dif)
        true_x_difs = true_x_difs - approximate_difs


    # integrate every basis function
    encodings = torch.zeros(xs.shape[0], len(basis_functions), output_size, device=device)
    for i in range(len(basis_functions)):
        model = basis_functions[i]
        x_difs = integrate(model, xs, t_dif, absolute=False)
        individual_encodings = true_x_difs * x_difs
        individual_encodings = torch.mean(individual_encodings, dim=1)
        encodings[:, i, :] = individual_encodings

    return encodings * 1e3

def get_encodings_trajectory(basis_functions, wind_models, winds):
    # create trajectory
    init_states = (torch.rand(winds.shape[0], output_size) * (x_range[1] - x_range[0]) + x_range[0]).to(device)
    xs, ts = gather_trajectories(wind_models, winds, init_states)
    t_difs = ts[1:] - ts[:-1]

    # get true x difs
    true_x_difs = (xs[:, 1:, :] - xs[:, :-1, :])
    if use_approximate_model:
        approximate_difs = approximate_model(xs[:, :-1, :], t_difs)
        true_x_difs = true_x_difs - approximate_difs

    # integrate
    encodings = torch.zeros(xs.shape[0], len(basis_functions), output_size, device=device)
    for i in range(len(basis_functions)):
        model = basis_functions[i]
        x_difs = integrate(model, xs[:, :-1, :], t_difs, absolute=False)
        individual_encodings = true_x_difs * x_difs
        individual_encodings = torch.mean(individual_encodings, dim=1)
        encodings[:, i, :] = individual_encodings

    return encodings * 1e3

def get_encodings_trajectory_ls(basis_functions, wind_models, winds):
    # create trajectory
    init_states = (torch.rand(winds.shape[0], output_size) * (x_range[1] - x_range[0]) + x_range[0]).to(device)
    xs, ts = gather_trajectories(wind_models, winds, init_states)
    t_difs = ts[1:] - ts[:-1]

    # integrate
    F = (xs[:, 1:, :] - xs[:, :-1, :])
    if use_approximate_model:
        approximate_difs = approximate_model(xs[:, :-1, :], t_difs)
        F = F - approximate_difs


    G = torch.zeros(F.shape[0],F.shape[1], F.shape[2], len(basis_functions), device=device)

    for i in range(len(basis_functions)):
        model = basis_functions[i]
        x_difs = integrate(model, xs[:, :-1, :], t_difs, absolute=False)
        G[:, :, :, i] = x_difs

    GTG = torch.einsum("fxnk,fxnc->fnkc", G, G)
    GTF = torch.einsum("fxnk,fxn->fnk", G, F)
    GTG_inv = torch.inverse(GTG + torch.eye(GTG.shape[-1], device=device))
    C = torch.einsum("fnkk,fnk->fnk", GTG_inv, GTF).transpose(1,2)

    return C

def predict(basis_functions, encodings, xs, t_difs):
    # define a model for the weight sum of basis functions multiplied by encodings
    def model(x):
        basis_delta_xs = torch.concat([basis_functions[i](x).unsqueeze(1) for i in range(len(basis_functions))], dim=1)
        return torch.einsum("fkdx,fkx->fdx", basis_delta_xs, encodings)

    # integrate the model
    outs = xs + integrate(model, xs, t_difs, absolute=False)
    if use_approximate_model:
        outs = outs + approximate_model(xs, t_difs)
    return outs

# create a neural ode function encoder
basis_functions = [torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        ).to(device) for k in range(embed_size)]
optimizer = torch.optim.Adam(chain(*[m.parameters() for m in basis_functions]), lr=1e-3)

losses = []
for step in trange(descent_steps):
    # gather data
    wind_models, winds = get_windy_models(num_functions, wind_range)
    init_states = (torch.rand(num_functions, 4) * (x_range[1] - x_range[0]) + x_range[0]).to(device)
    init_states[:, 2:] = 0.0
    xs, ts = gather_trajectories(wind_models, winds, init_states)

    # compute coefficients
    encodings = get_encodings(basis_functions, wind_models, winds, type=train_type)

    # compute loss
    t_difs = ts[1:] - ts[:-1]
    y_hat = predict(basis_functions, encodings, xs[:, :-1, :], t_difs)
    loss = torch.nn.MSELoss()(y_hat, xs[:, 1:, :])

    # backprop
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(chain(*[m.parameters() for m in basis_functions]), 1)
    optimizer.step()
    losses.append(loss.item())

    # logs
    logger.add_scalar("loss", loss.item(), step)
    logger.add_scalar("grad_norm", norm, step)


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
# plot
# creates 9 models for 1000 time steps
windy_models, winds = get_windy_models(9, wind_range)
xs = torch.zeros(9, 1000, 4, device=device)
xs_estimated = torch.zeros(9, 1000, 4, device=device)
xs[:, 0, 0:2] = torch.rand(9, 2, device=device) * (x_range[1] - x_range[0]) + x_range[0]
xs_estimated[:, 0, :] = xs[:, 0, :]

# get encoding
encodings = get_encodings(basis_functions, windy_models, winds,  type=test_type)


# create video
width, height = 1000, 1000
out = cv2.VideoWriter(f'{logdir}/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
fig, axs = plt.subplots(3, 3)
fig.set_size_inches(width / 100, height / 100)
fig.set_dpi(100)
fig.tight_layout()
for i in range(9):
    axs[i // 3, i % 3].set_title(f"wind={winds[i, 0]:.2f},{winds[i, 1]:.2f}")
    axs[i//3, i%3].set_xlim(-10, 10)
    axs[i//3, i%3].set_ylim(-10, 10)

# run ploter
for time_index in trange(plot_time-1):
    # run the vanderpol model
    x_now = xs[:, time_index, :]
    t_dif = torch.tensor(0.1).to(device)
    x_next = windy_models(x_now, t_dif)
    xs[:, time_index+1, :] = x_next

    # run the fe model
    x_now = xs_estimated[:, time_index:time_index+1, :]
    x_next = predict(basis_functions, encodings, x_now, t_dif)
    xs_estimated[:, time_index+1, :] = x_next.squeeze(1)

    # plot the new point
    for i in range(9):
        axs[i//3, i%3].plot([xs[i, time_index,  0].cpu().detach().numpy(), xs[i,time_index+1, 0].cpu().detach().numpy()],
                            [xs[i, time_index,  1].cpu().detach().numpy(), xs[i,time_index+1, 1].cpu().detach().numpy()],
                            color="black")
        axs[i//3, i%3].plot([xs_estimated[i, time_index,  0].cpu().detach().numpy(), xs_estimated[i,time_index+1, 0].cpu().detach().numpy()],
                            [xs_estimated[i, time_index,  1].cpu().detach().numpy(), xs_estimated[i,time_index+1, 1].cpu().detach().numpy()],
                            color="red", marker="o", markersize=2)


    # Convert graph to numpy image
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
    out.write(img)
out.release()