import os
from itertools import chain
import matplotlib.cm as cmx

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import colors, ticker
from matplotlib.gridspec import GridSpec
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
mu_range = [0.1, 3.0] # [1, 2]
x_range = [-2.0, 2.0]
y_range = [-2.0, 2.0]
t_range = [0, 10]
num_functions = 10

# nn params
input_size = 2
hidden_size = 100
output_size = 2
device = "cuda:0"
embed_size = 11
descent_steps = args.grad_steps
plot_time = args.plot_time
use_approximate_model = args.use_approximate_model
approximate_model_mu = 1.0
train_type = args.train_type
test_type = args.test_type

train = False # if not train, uses most recent
video_fit = True
video_interpolate = False
image_interpolate = False
images_flow = False
theta_search = False

assert train_type in ["uniform", "trajectory", "trajectory_ls"]
assert test_type in ["uniform", "trajectory", "trajectory_ls"]



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

def get_vanderpol_models(number_models, mu_range, absolute=True):
    mus = torch.rand(number_models, device=device) * (mu_range[1] - mu_range[0]) + mu_range[0]
    def vanderpol(x):
        if len(x.shape) == 2:
            dx = x[:, 1]
            dy = mus * (1 - x[:, 0]**2) * x[:, 1] - x[:, 0]
            return torch.concat([dx.unsqueeze(1), dy.unsqueeze(1)], dim=1)
        else:
            dx = x[:, :, 1]
            dy = mus.unsqueeze(1) * (1 - x[:, :, 0]**2) * x[:, :, 1] - x[:, :, 0]
            return torch.concat([dx.unsqueeze(2), dy.unsqueeze(2)], dim=2)

    return lambda x_0, t_dif: integrate(vanderpol, x_0, t_dif, absolute=absolute), mus

def get_specific_vanderpol_models(number_models, mus, absolute=True):
    mus = torch.tensor(mus, device=device)
    def vanderpol(x):
        if len(x.shape) == 2:
            dx = x[:, 1]
            dy = mus * (1 - x[:, 0]**2) * x[:, 1] - x[:, 0]
            return torch.concat([dx.unsqueeze(1), dy.unsqueeze(1)], dim=1)
        else:
            dx = x[:, :, 1]
            dy = mus.unsqueeze(1) * (1 - x[:, :, 0]**2) * x[:, :, 1] - x[:, :, 0]
            return torch.concat([dx.unsqueeze(2), dy.unsqueeze(2)], dim=2)

    return lambda x_0, t_dif: integrate(vanderpol, x_0, t_dif, absolute=absolute), mus

# the approximate model if it is used
approximate_model = get_vanderpol_models(1, [approximate_model_mu,approximate_model_mu], absolute=False)[0]

def gather_trajectories(model, mus, x_0, t_length=10.0, num_points=1000):
    # random timepoints between 0 and length
    timepoints = torch.sort(torch.rand(num_points) * t_length).values
    ts = torch.cat([torch.tensor([0.0]), timepoints]).to(device)

    # initialize space for states
    xs = torch.zeros(mus.shape[0], num_points + 1, 2, device=device)
    xs[:, 0, :] = x_0

    # gather data
    for i in range(num_points):
        t_dif = ts[i+1] - ts[i]
        x_now = xs[:, i, :]
        x_next = model(x_now, t_dif)
        xs[:, i+1, :] = x_next
        if torch.isnan(x_next).any():
            print("nan")
            x_next = model(x_now, t_dif)
    return xs, ts

def get_encodings(basis_functions, vanderpol_models, mus, type="uniform"):
    if type == "uniform":
        return get_encoding_uniform(basis_functions, vanderpol_models, mus)
    elif type == "trajectory":
        return get_encodings_trajectory(basis_functions, vanderpol_models, mus)
    elif type == "trajectory_ls":
        return get_encodings_trajectory_ls(basis_functions, vanderpol_models, mus)
    else:
        raise ValueError(f"Invalid type '{type}'")


def get_encoding_uniform(basis_functions, vanderpol_models, mus):
    xs = torch.rand(mus.shape[0], 1000, output_size, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    t_dif = torch.tensor(0.1).to(device)
    xs_next = vanderpol_models(xs, t_dif)


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

def get_encodings_trajectory(basis_functions, vanderpol_models, mus):
    # create trajectory
    init_states = (torch.rand(mus.shape[0], output_size) * (x_range[1] - x_range[0]) + x_range[0]).to(device)
    xs, ts = gather_trajectories(vanderpol_models, mus, init_states)
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
        if torch.isnan(encodings).any():
            print("nan")
    return encodings * 1e3

def get_encodings_trajectory_ls(basis_functions, vanderpol_models, mus):
    # create trajectory
    init_states = (torch.rand(mus.shape[0], output_size) * (x_range[1] - x_range[0]) + x_range[0]).to(device)
    xs, ts = gather_trajectories(vanderpol_models, mus, init_states)
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
if train:
    # add learning rate decay
    decay_rate = 0.999
    optimizer = torch.optim.Adam(chain(*[m.parameters() for m in basis_functions]), lr=1e-3)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = f"logs/vanderpol/{date_time_str}/{train_type}_{test_type}{'_approx' if use_approximate_model else ''}"
    os.makedirs(logdir, exist_ok=True)
    logger = SummaryWriter(logdir)

    losses = []
    for step in trange(descent_steps):
        # gather data
        vanderpol_models, mus = get_vanderpol_models(num_functions, mu_range)
        init_states = (torch.rand(num_functions, 2) * (x_range[1] - x_range[0]) + x_range[0]).to(device)
        xs, ts = gather_trajectories(vanderpol_models, mus, init_states)

        # compute coefficients
        encodings = get_encodings(basis_functions, vanderpol_models, mus, type=train_type)

        # compute loss
        t_difs = ts[1:] - ts[:-1]
        y_hat = predict(basis_functions, encodings, xs[:, :-1, :], t_difs)
        loss = torch.nn.MSELoss()(y_hat, xs[:, 1:, :])

        # compute loss by integrating a full trajectory
        # t_difs = ts[1:] - ts[:-1]
        # y_hats = []
        # y_current = xs[:, 0, :]
        # for i in range(1, xs.shape[1]):
        #     y_next = predict(basis_functions, encodings, y_current.unsqueeze(1), t_difs[i-1]).squeeze(1)
        #     y_hats.append(y_next)
        #     y_current = y_next
        # y_hats_tensor = torch.stack(y_hats, dim=1)
        # assert y_hats_tensor.shape == xs[:, 1:, :].shape
        # loss = torch.nn.MSELoss()(y_hats_tensor, xs[:, 1:, :])

        # backprop
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(chain(*[m.parameters() for m in basis_functions]), 1)
        optimizer.step()
        losses.append(loss.item())

        # logs
        logger.add_scalar("loss", loss.item(), step)
        logger.add_scalar("grad_norm", norm, step)

        my_lr_scheduler.step()

    for basis_index in range(len(basis_functions)):
        torch.save(basis_functions[basis_index].state_dict(), f"{logdir}/basis_{basis_index}.pt")
else: # load model
    # get latest directory in logs/vanderpol
    dirs = os.listdir("logs/vanderpol")
    newest = max(dirs, key=lambda x: os.path.getctime(f"logs/vanderpol/{x}"))
    logdir = f"logs/vanderpol/{newest}/{train_type}_{test_type}{'_approx' if use_approximate_model else ''}"
    for i in range(embed_size):
        basis_functions[i].load_state_dict(torch.load(f"{logdir}/basis_{i}.pt"))



################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
# plot
# creates 9 models for 1000 time steps
if video_fit:
    n_rows, n_cols = 2, 4
    vanderpol_models, mus = get_vanderpol_models(n_rows * n_cols, mu_range)
    xs = torch.zeros(n_rows * n_cols, 1000, 2, device=device)
    xs_estimated = torch.zeros(n_rows * n_cols, 1000, 2, device=device)
    xs[:, 0, :] = torch.rand(n_rows * n_cols, 2, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    xs_estimated[:, 0, :] = xs[:, 0, :]

    # get encoding
    encodings = get_encodings(basis_functions, vanderpol_models, mus,  type=test_type)


    # create video
    width, height = n_cols * 600, n_rows * 600
    out = cv2.VideoWriter(f'{logdir}/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    fig, axs = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(width / 100, height / 100)
    fig.set_dpi(100)
    fig.tight_layout(h_pad=3.0, rect=[0, 0, 1, 0.98])
    for i in range(n_rows * n_cols):
        axs[i//n_cols, i%n_cols].set_xlim(-5, 5)
        axs[i//n_cols, i%n_cols].set_ylim(-5, 5)
        axs[i//n_cols, i%n_cols].tick_params(axis='both', which='major', labelsize=18)
        axs[i//n_cols, i%n_cols].set_title(f"$\mu={mus[i]:.2f}$", fontsize=24)

    # run ploter
    for time_index in trange(plot_time-1):
        # run the vanderpol model
        x_now = xs[:, time_index, :]
        t_dif = torch.tensor(0.1).to(device)
        x_next = vanderpol_models(x_now, t_dif)
        xs[:, time_index+1, :] = x_next

        # run the fe model
        x_now = xs_estimated[:, time_index:time_index+1, :]
        x_next = predict(basis_functions, encodings, x_now, t_dif)
        xs_estimated[:, time_index+1, :] = x_next.squeeze(1)

        # plot the new point
        for i in range(n_rows * n_cols):
            axs[i//n_cols, i%n_cols].plot([xs[i, time_index,  0].cpu().detach().numpy(), xs[i,time_index+1, 0].cpu().detach().numpy()],
                                [xs[i, time_index,  1].cpu().detach().numpy(), xs[i,time_index+1, 1].cpu().detach().numpy()],
                                color="black")
            axs[i//n_cols, i%n_cols].plot([xs_estimated[i, time_index,  0].cpu().detach().numpy(), xs_estimated[i,time_index+1, 0].cpu().detach().numpy()],
                                [xs_estimated[i, time_index,  1].cpu().detach().numpy(), xs_estimated[i,time_index+1, 1].cpu().detach().numpy()],
                                color="red", marker="o", markersize=2)


        # Convert graph to numpy image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(data, cv2.COLOR_RGBA2BGR)
        out.write(img)
    out.release()

if video_interpolate:
    # now compute the representations for vanderpol models of 0.5, 2.0, and show the linear combinations of their representations
    # creates 2 models
    vanderpol_model1, mu1 = get_vanderpol_models(1, [1.0, 1.0])
    vanderpol_model2, mu2 = get_vanderpol_models(1, [2.0, 2.0])


    # get encoding
    encoding1 = get_encodings(basis_functions, vanderpol_model1, mu1,  type=test_type)
    encoding2 = get_encodings(basis_functions, vanderpol_model2, mu2,  type=test_type)

    # create all 9 encodings
    encodings = []
    mus = []
    # instead of theta going from 0 to 1, make theta go from -1 to 2
    for i in range(9):
        theta =  1 - (i / 4)
        encodings.append(encoding1 * (theta) + encoding2 * (1 - theta))
        mus.append((mu1 * (theta) + mu2 * (1 - theta)).item())
    encodings = torch.stack(encodings, dim=0).squeeze(1)

    # now create all 9 mus
    vanderpol_models, mus = get_specific_vanderpol_models(9, mus, absolute=True)

    # Now plot everything
    xs_estimated = torch.zeros(9, 1000, 2, device=device)
    xs_estimated[:, 0, :] = xs[:, 0, :]


    # create video
    width, height = 1000, 1000
    out = cv2.VideoWriter(f'{logdir}/output_interpolate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    fig, axs = plt.subplots(3, 3)
    fig.set_size_inches(width / 100, height / 100)
    fig.set_dpi(100)
    fig.tight_layout()
    for i in range(9):
        approx_mu =  1 + i/4
        title = f"Approximation of mu={approx_mu}"
        axs[i // 3, i % 3].set_title(title)
        axs[i//3, i%3].set_xlim(-5, 5)
        axs[i//3, i%3].set_ylim(-5, 5)

    # run ploter
    for time_index in trange(plot_time-1):
        # run the vanderpol model
        x_now = xs[:, time_index, :]
        t_dif = torch.tensor(0.1).to(device)
        x_next = vanderpol_models(x_now, t_dif)
        xs[:, time_index+1, :] = x_next

        # run the fe model
        x_now = xs_estimated[:, time_index:time_index+1, :]
        x_next = predict(basis_functions, encodings, x_now, t_dif)
        xs_estimated[:, time_index+1, :] = x_next.squeeze(1)

        # plot the new point
        for i in range(9):
            axs[i // 3, i % 3].plot(
                [xs[i, time_index, 0].cpu().detach().numpy(), xs[i, time_index + 1, 0].cpu().detach().numpy()],
                [xs[i, time_index, 1].cpu().detach().numpy(), xs[i, time_index + 1, 1].cpu().detach().numpy()],
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
    plt.clf()

# create single rendering of different values of mu vs the interpolated estimate
# create a 1 row, 2 column matplotlib plot with the second column as a color bar and tiny
if image_interpolate:
    fig = plt.figure(figsize=(2.1*3, 1 * 3))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # create a color bar
    cmap_type = 'viridis'
    jet = plt.get_cmap(cmap_type)
    cNorm = colors.Normalize(vmin=1, vmax=3)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    # make last plot the colorbar
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(scalarMap, cax=cax)

    # create ground truth vanderpol systems
    mus = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    vanderpol_models, mus = get_specific_vanderpol_models(9, mus, absolute=True)

    # create space to store the ground truth and the estimate
    horizon = 10_000 # long enough to reach steady state
    xs = torch.zeros(len(mus), horizon, 2, device=device)
    xs_estimated = torch.zeros(len(mus), horizon, 2, device=device)
    xs[:, 0, :] = torch.rand(len(mus), 2, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    xs_estimated[:, 0, :] = xs[:, 0, :]

    # estimate ground truth
    for time_index in trange(horizon-1):
        x_now = xs[:, time_index, :]
        t_dif = torch.tensor(0.1).to(device)
        x_next = vanderpol_models(x_now, t_dif)
        xs[:, time_index+1, :] = x_next

    # get representaiton for mu = 1.0 and mu = 2.0
    vanderpol_model1, mu1 = get_vanderpol_models(1, [1.0, 1.0])
    vanderpol_model2, mu2 = get_vanderpol_models(1, [2.0, 2.0])

    # get encoding
    encoding1 = get_encodings(basis_functions, vanderpol_model1, mu1,  type=test_type)
    encoding2 = get_encodings(basis_functions, vanderpol_model2, mu2,  type=test_type)

    # create all 9 encodings
    encodings = []
    mus2 = []
    # instead of theta going from 0 to 1, make theta go from -1 to 2
    for i in range(9):
        theta =  1 - (i / 4)
        encodings.append(encoding1 * (theta) + encoding2 * (1 - theta))
        mus2.append((mu1 * (theta) + mu2 * (1 - theta)).item())
    encodings = torch.stack(encodings, dim=0).squeeze(1)
    print(mus2, mus)

    # now simulate all encodings
    for time_index in trange(horizon-1):
        # run the fe model
        x_now = xs_estimated[:, time_index:time_index+1, :]
        x_next = predict(basis_functions, encodings, x_now, t_dif)
        xs_estimated[:, time_index+1, :] = x_next.squeeze(1)


    # plot the ground truth in the first ax. Only care about the last 1000 points
    for i in range(len(mus)):
        color = scalarMap.to_rgba(mus2[i])
        ax1.plot(xs[i, -1000:, 0].cpu().detach().numpy(), xs[i, -1000:, 1].cpu().detach().numpy(), color=color)

    # plot the estimate in the second ax. Only care about the last 1000 points
    for i in range(len(mus)):
        color = scalarMap.to_rgba(mus2[i])
        ax2.plot(xs_estimated[i, -1000:, 0].cpu().detach().numpy(), xs_estimated[i, -1000:, 1].cpu().detach().numpy(), color=color)


    # set labels, titles
    ax1.set_title("Ground Truth")
    ax2.set_title("Estimated")

    # set xlims
    ax1.set_xlim(-2.3, 2.3)
    ax2.set_xlim(-2.3, 2.3)
    ax1.set_ylim(-5, 5)
    ax2.set_ylim(-5, 5)

    # set colorbar labels
    cb.locator = ticker.LinearLocator(9) # ticker.MaxNLocator(nbins=3)
    cb.update_ticks()
    tick_labels = [f"{i:.2f}{'*' if i == 1.0 or i == 2.0 else ''}" for i in mus2]
    cb.ax.set_yticklabels(tick_labels)

    # save it to image
    plt.savefig(f"{logdir}/ground_truth_vs_estimate.png")

if images_flow:
    # define the model
    def model_x_dot(x, encodings):
        basis_delta_xs = torch.concat([basis_functions[i](x).unsqueeze(1) for i in range(len(basis_functions))], dim=1)
        return torch.einsum("fkdx,fkx->fdx", basis_delta_xs, encodings)

    # define a bunch of mus and interpolated encodings
    vanderpol_model1, mu1 = get_vanderpol_models(1, [1.0, 1.0])
    vanderpol_model2, mu2 = get_vanderpol_models(1, [2.0, 2.0])

    # get encoding
    encoding1 = get_encodings(basis_functions, vanderpol_model1, mu1, type=test_type)
    encoding2 = get_encodings(basis_functions, vanderpol_model2, mu2, type=test_type)

    # create all 9 encodings
    encodings = []
    mus2 = []
    # instead of theta going from 0 to 1, make theta go from -1 to 2
    for i in range(9):
        theta = 1 - (i / 4)
        encodings.append(encoding1 * (theta) + encoding2 * (1 - theta))
        mus2.append((mu1 * (theta) + mu2 * (1 - theta)).item())
    encodings = [encodings[0], encodings[2], encodings[4], encodings[8]]
    mus = [mus2[0], mus2[2], mus2[4], mus2[8]]

    for encoding, mu in zip(encodings, mus):
        # Plot a streamplot of the vector vield of a van der Pol oscillator
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        xmin = -3
        xmax = 3
        ymin = -5
        ymax = 5
        density = 100

        # generate input data
        XX, YY = np.meshgrid(np.linspace(xmin, xmax, density), np.linspace(ymin, ymax, density))
        XX_tensor = torch.tensor(XX, device=device).float().flatten()
        YY_tensor = torch.tensor(YY, device=device).float().flatten()

        # get encodings for a given mu
        vanderpol_model, mu = get_vanderpol_models(1, [mu, mu])

        # compute the vector field
        inputs = torch.cat([XX_tensor.unsqueeze(1), YY_tensor.unsqueeze(1)], dim=1).unsqueeze(0)
        x_dots = model_x_dot(inputs, encoding).reshape(density, density, 2)
        U = x_dots[:, :, 0].cpu().detach().numpy()
        V = x_dots[:, :, 1].cpu().detach().numpy()

        # plot the stream
        ax.streamplot(XX, YY, U, V, color='0.5', linewidth=0.5, density=1.0, broken_streamlines=False)

        # simulate a single trajectory using the encodings and plot it
        x_range = [-4.0, 4.0]
        x = torch.rand(1, 1, 2, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        xs = [x.squeeze(0).squeeze(0)]
        t_dif = torch.tensor(0.05).to(device)
        for _ in range(1000):
            x = predict(basis_functions, encoding, x, t_dif)
            xs.append(x.squeeze(0).squeeze(0))

        xs = torch.stack(xs, dim=0).squeeze(1).cpu().detach().numpy()
        ax.plot(xs[:, 0], xs[:, 1], color="blue", linewidth=3)


        # do lims
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylim(ymin=ymin, ymax=ymax)

        # remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # reformat
        plt.tight_layout()
        ax.tick_params(axis=u'both', which=u'both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)


        plt.savefig(f"{logdir}/vector_field_mu_{mu.item():.2f}.pdf", bbox_inches="tight", pad_inches=0.0)
        # plt.show()


if theta_search:
    # define a bunch of mus and interpolated encodings
    vanderpol_model1, mu1 = get_vanderpol_models(1, [1.0, 1.0])
    vanderpol_model2, mu2 = get_vanderpol_models(1, [2.0, 2.0])

    # get encoding
    encoding1 = get_encodings(basis_functions, vanderpol_model1, mu1, type=test_type)
    encoding2 = get_encodings(basis_functions, vanderpol_model2, mu2, type=test_type)


    # unseen mu
    unseen_mu = 3
    num_data_points_observed = 200

    # get a trajectory from unseen mu
    vanderpol_model_unseen, mu_unseen = get_vanderpol_models(1, [unseen_mu, unseen_mu])
    init_state = (torch.rand(1, 2) * (x_range[1] - x_range[0]) + x_range[0]).to(device)
    xs, ts = gather_trajectories(vanderpol_model_unseen, mu_unseen, init_state, t_length=20.0, num_points=2000)
    example_states = xs[:, :num_data_points_observed, :]
    example_next_states = xs[:, 1:num_data_points_observed+1, :]
    future_data = xs[:, num_data_points_observed+1:, :]
    t_difs = ts[1:] - ts[:-1]
    t_difs_examples = t_difs[:num_data_points_observed]
    t_difs_future = t_difs[num_data_points_observed:]


    # predict transitions using the encodings
    predictions1 = predict(basis_functions, encoding1, example_states, t_difs_examples) - example_states
    predictions2 = predict(basis_functions, encoding2, example_states, t_difs_examples) - example_states

    # find optimal mu via gradient descent
    mu_estimate = torch.tensor(0.5, requires_grad=True, device=device)
    opti = torch.optim.Adam([mu_estimate], lr=1e-3)
    pbar = trange(100000)
    for descent_step in pbar:
        approximate_next_state = predictions1.detach() * mu_estimate + predictions2.detach() * (1 - mu_estimate) + example_states.detach()
        loss = torch.nn.MSELoss()(approximate_next_state, example_next_states.detach())
        loss.backward()
        opti.step()
        opti.zero_grad()
        pbar.set_description(f"l={loss.item()}, mu={mu_estimate.item():.2f}")
    print(loss.item())
    print(mu_estimate.item())
    # now compute the approximate coefficients
    encoding_estimate = encoding1 * mu_estimate + encoding2 * (1 - mu_estimate)

    # now predict the future of the trajectory
    states_predicted_by_estimator = torch.zeros_like(xs)
    states_predicted_by_estimator[:, :num_data_points_observed+1, :] = xs[:, :num_data_points_observed+1, :]
    for time in range(num_data_points_observed, states_predicted_by_estimator.shape[1]-1):
        states_predicted_by_estimator[:, time+1, :] = predict(basis_functions, encoding_estimate, states_predicted_by_estimator[:, time:time+1, :], t_difs[time:time+1])

    # now plot the true trajectory and the estimated trajectory
    fig, ax = plt.subplots(1, 1)
    # first plot the point seen in black
    ax.plot(xs.squeeze(0)[:num_data_points_observed+1, 0].cpu().detach().numpy(), xs.squeeze(0)[:num_data_points_observed+1, 1].cpu().detach().numpy(), color="black")
    # then plot the future in gray
    ax.plot(xs.squeeze(0)[num_data_points_observed:, 0].cpu().detach().numpy(), xs.squeeze(0)[num_data_points_observed:, 1].cpu().detach().numpy(), color="gray")
    # then plot the estimated future in red
    ax.plot(states_predicted_by_estimator.squeeze(0)[num_data_points_observed:, 0].cpu().detach().numpy(), states_predicted_by_estimator.squeeze(0)[num_data_points_observed:, 1].cpu().detach().numpy(), color="red")
    plt.savefig(f"{logdir}/theta_search_n={num_data_points_observed}.pdf")

