import os

import matplotlib.pyplot as plt
import torch

plot_trajs = True
plot_filtered_mses = True

env = 0
if plot_trajs:
    alg = "MLP"
    mses = torch.load(f"logs/drone/predictor/mses.pt")
    mses = mses[alg][0]
    for env in range(mses.shape[0]):
        for episode in range(mses.shape[1]):
            plt.plot(mses[env, episode, :].cpu().detach().numpy(), alpha=0.5, color="black")
        plt.plot(torch.mean(mses[env, :, :], dim=0).cpu().detach().numpy(), color="red")
        plt.title(f"Env {env}")
        plt.ylim(0, 5)
        plt.show()
