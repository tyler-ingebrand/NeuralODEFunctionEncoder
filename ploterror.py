import os

import matplotlib.pyplot as plt
import torch

plot_trajs = False
plot_filtered_mses = True


if plot_trajs:
    alg = "FE"
    mses = torch.load(f"/tmp/{alg}_mse.pt")
    for env in range(mses.shape[0]):
        for episode in range(mses.shape[1]):
            plt.plot(mses[env, episode, :].cpu().detach().numpy(), alpha=0.5, color="black")
        plt.plot(torch.mean(mses[env, :, :], dim=0).cpu().detach().numpy(), color="red")
        plt.title(f"Env {env}")
        plt.show()


if plot_filtered_mses:
    mses = {}
    algs = ["MLP", "NeuralODE", "FE", "FE_NeuralODE", "FE_Residuals", "FE_NeuralODE_Residuals", "Zeros"]
    max_mse = 5
    for alg in algs:
            mse = torch.load(f"/tmp/{alg}_mse.pt")

            # filter out trajectories with high mse
            large_mse = mse > max_mse
            trajs_to_remove = torch.any(large_mse, dim=2)

            # remove the rows with high mse
            mses_filtered = []
            for i in range(mse.shape[0]):
                for j in range(mse.shape[1]):
                    if not trajs_to_remove[i, j]:
                        mses_filtered.append(mse[i, j, :])
            mses_filtered = torch.stack(mses_filtered)


            mse = torch.mean(mses_filtered, dim=(0))
            mses[alg] = mse

    # plot using matplotlib
    plt.figure()
    for alg_type, mse in mses.items():
        plt.plot(mse.cpu().detach().numpy(), label=alg_type)

    # plot a veritcal line at x=2
    # plt.axvline(x=3, color="black", linestyle="--")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("MSE over time horizon")
    plt.xlabel("Time step")
    plt.ylabel("MSE")
    plt.ylim(0, 1.0)
    # plt.show()
    save_dir = "logs/cheetah/predictor/"
    plt.savefig(os.path.join(save_dir , "filtered_mse_over_time_horizon.png"), bbox_inches="tight")
