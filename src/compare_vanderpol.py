import os

import torch

from src.vanderpol import get_vanderpol_models, get_encodings, predict_node, predict, get_specific_vanderpol_models
import matplotlib.pyplot as plt
import numpy as np

with torch.no_grad():
    plt.rcParams["font.family"] = "Times New Roman"
    # hyper params
    logdir = "logs/vanderpol"
    dirs = ["2024-04-12_11-27-35",
            "2024-04-12_15-08-58",
            # "2024-04-30_13-05-44",
            ]
    input_size = 2
    hidden_size = 100
    output_size = 2
    device = "cuda:0"
    embed_size = 11
    mu_range = [0.1, 3.0] # [1, 2]
    x_range = [-2.0, 2.0]
    y_range = [-2.0, 2.0]
    t_range = [0, 10]
    time_horizon = 1000

    # load all models into memory
    models = {}
    for dir in dirs:
        # load the subdirectories
        subdirs = [x for x in os.listdir(os.path.join(logdir, dir)) if os.path.isdir(os.path.join(logdir, dir, x))]
        assert len(subdirs) == 1, f"Expected 1 subdir, got {len(subdirs)}"

        # fetch model type
        if "node" in subdirs[0]:
            alg_type = "neuralODE"
        elif "approx" in subdirs[0]:
            alg_type = "FE_NeuralODE_Residual"
        else:
            alg_type = "FE_NeuralODE"

        # load the model
        alg_dir = os.path.join(logdir, dir, subdirs[0])
        if alg_type == "neuralODE":
            model = neural_ode = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            ).to(device)
            model.load_state_dict(torch.load(os.path.join(alg_dir, "node.pt")))
        else:
            model = [torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size),
            ).to(device) for k in range(embed_size)]
            for k in range(embed_size):
                model[k].load_state_dict(torch.load(os.path.join(alg_dir, f"basis_{k}.pt")))

        models[alg_type] = model


    # now simulate 9 systems and see how each approach performs.
    n_vanderpols = 8
    num_rows = 2
    num_cols = 4
    mus = torch.linspace(mu_range[0], mu_range[1], n_vanderpols, device=device)    # torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=device)
    vanderpol_models, mus = get_specific_vanderpol_models(len(mus), mus)
    xs = torch.zeros(n_vanderpols, time_horizon, 2, device=device)
    xs_estimated = torch.zeros(len(models), n_vanderpols, time_horizon, 2, device=device)
    xs[:, 0, :] = torch.rand(n_vanderpols, 2, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    xs_estimated[:, :, 0, :] = xs[:, 0, :]

    # get encoding
    encodings = {}
    for alg_type, model in models.items():
        if alg_type != "neuralODE":
            encodings[alg_type] = get_encodings(model, vanderpol_models, mus,  type="trajectory")

    # simulate
    for time_index in range(time_horizon-1):
        # run the vanderpol model
        x_now = xs[:, time_index, :]
        t_dif = torch.tensor(0.02).to(device)
        x_next = vanderpol_models(x_now, t_dif)
        xs[:, time_index+1, :] = x_next

        # run the learned models
        for i, alg_type in enumerate(models.keys()):
            x_now = xs_estimated[i, :, time_index, :]
            model = models[alg_type]
            if alg_type == "neuralODE":
                x_next = predict_node(model, x_now, t_dif)
            else:
                x_next = predict(model, encodings[alg_type], x_now.unsqueeze(1), t_dif).squeeze(1)
            xs_estimated[i, :, time_index+1, :] = x_next

    # plot
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*3, num_rows*3))
    legend_labels = {}
    for i in range(n_vanderpols):
        row = i // num_cols
        col = i % num_cols
        l1, = ax[row, col].plot(xs[i, :, 0].cpu().numpy(), xs[i, :, 1].cpu().numpy(), label="Ground Truth", color="black", linestyle="--")
        legend_labels["Ground Truth"] = l1
        for j, alg_type in enumerate(models.keys()):
            color = "red" if alg_type == "neuralODE" else "blue"
            label = "NODE" if alg_type == "neuralODE" else "FE + NODE + Res."

            l, = ax[row, col].plot(xs_estimated[j, i, :, 0].cpu().numpy(), xs_estimated[j, i, :, 1].cpu().numpy(), label=label, color=color)
            legend_labels[label] = l

        ax[row, col].set_title(f"$\mu$={mus[i]:.2f}", y=1.0,  loc="left", pad=-14, x=0.05)
        # ax.set_title('Manual y', y=1.0, pad=-14)
        ax[row, col].set_ylim(-5.5,6)

        # no y ticks for the right columns
        if col != 0:
            ax[row, col].set_yticks([])
        if row != num_rows-1:
            ax[row, col].set_xticks([])
    # order = ["NODE", "FE + Res.", "FE + NODE", "FE + NODE + Res.", "Oracle"]
    # legend_labels = {k: legend_labels[k] for k in order}

    fig.legend(loc=10,
               labels=legend_labels.keys(), handles=legend_labels.values(),
               ncol=5,
               bbox_to_anchor=(0.5, 0.02),
               edgecolor=(1, 1, 1, 0.), facecolor=(1, 1, 1, 0.0)
               )

    plt.tight_layout(rect=(-0.01, 0.02, 1, 1))
    plt.savefig(f'{logdir}/vanderpol_comparison.pdf', dpi=300)

    # save to csv
    n_rows = time_horizon
    n_cols = 4 * n_vanderpols # x and y for true and estimated, for all 9 vanderpols

    for j, alg_type in enumerate(models.keys()):
        data = np.zeros((n_rows, n_cols))
        for i in range(n_vanderpols):
            data[:, i*4] = xs[i, :, 0].cpu().numpy()
            data[:, i*4+1] = xs[i, :, 1].cpu().numpy()
            data[:, i*4+2] = xs_estimated[j, i, :, 0].cpu().numpy()
            data[:, i*4+3] = xs_estimated[j, i, :, 1].cpu().numpy()

        # save to csv
        header = "x_true,y_true,x_estimated,y_estimated," * 9
        np.savetxt(f'{logdir}/vanderpol_comparison_{alg_type}.csv', data, delimiter=",",
                   header=header)

        # also write mus
        np.savetxt(f'{logdir}/vanderpol_comparison_mus.csv', mus.cpu(), delimiter=",")