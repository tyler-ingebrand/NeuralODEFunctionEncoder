import numpy as np
import torch
from torch import jit
from torch import nn, optim
import os
import cv2
from tqdm import trange
import colorednoise as cn







# credit to here:
# https://github.com/homangab/gradcem/blob/master/mpc/gradcem.py

class CEM():  # jit.ScriptModule):
    def __init__(self, planning_horizon, opt_iters, samples, top_samples, env, device, action_means):
        super().__init__()
        self.set_env(env)
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K, self.top_K = samples, top_samples
        self.device = device
        self.action_space = self.env.action_space
        self.action_means = action_means.to(self.device)
        self.action_means = 0.07148927728325129 * torch.ones_like(self.action_means) # this is a empirically tested hover value
        self.prior_mu = None
        self.prior_std = None

    def set_env(self, env):
        self.env = env
        if self.env is not None:
            self.a_size = env.a_size

    # @jit.script_method
    def forward(self, batch_size, return_plan=False, return_plan_each_iter=False):
        # Here batch is strictly if multiple CEMs should be performed!
        B = batch_size
        improved = True # iCEM is supposedely better
        beta = 1 # for iCEM


        action_min = self.action_space[:, 0].to(self.device)
        action_max = self.action_space[:, 1].to(self.device)
        # need to make sure it fits the env...
        if self.prior_mu is None:
            action_space_stds = (action_max - action_min) / 5

            # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
            a_mu = torch.zeros(self.H, B, 1, self.a_size, device=self.device) + self.action_means
            a_std = torch.zeros(self.H, B, 1, self.a_size, device=self.device) + action_space_stds
            n_iters = self.opt_iters * 5
        else:
            a_mu = self.prior_mu
            a_std = self.prior_std
            n_iters = self.opt_iters

        plan_each_iter = []
        for index in range(n_iters):
            self.env.reset_state(B*self.K)
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            # Sample actions (T x (B*K) x A)
            if index == 0:
                if not improved:
                    noise = torch.randn(self.H, B, self.K, self.a_size, device=self.device)
                if improved:
                    noise = cn.powerlaw_psd_gaussian(beta, (B, self.K, self.a_size, self.H))
                    noise = noise.transpose(3, 0, 1, 2)
                    noise = torch.tensor(noise, device=self.device, dtype=torch.float32)

                actions = (a_mu + a_std * noise).view(self.H, B * self.K, self.a_size)

            else:
                # sample only action dims that were not top k
                if not improved:
                    noise = torch.randn(self.H, B, self.K - self.top_K, self.a_size, device=self.device)
                if improved:
                    noise = cn.powerlaw_psd_gaussian(beta, (B, self.K - self.top_K, self.a_size, self.H))
                    noise = noise.transpose(3, 0, 1, 2)
                    noise = torch.tensor(noise, device=self.device, dtype=torch.float32)
                actions[:, self.top_K:] = (a_mu + a_std * noise).view(self.H, B * (self.K - self.top_K), self.a_size)
            actions = torch.max(torch.min(actions, action_max), action_min)

            # Returns (B*K)
            with torch.no_grad():
                if index == n_iters - 1:
                    returns = self.env.rollout(actions, render=True) # optionally render a plan
                else:
                    returns = self.env.rollout(actions)

            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.K).topk(self.top_K, dim=1, largest=True, sorted=False)
            topk += self.K * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(self.H, B, self.top_K, self.a_size)

            # save best actions as top of actions
            # only do this if we iter again
            if index != self.opt_iters - 1:
                actions[:, :self.top_K] = best_actions.squeeze(1)


            # Update belief with new means and standard deviations
            a_mu = best_actions.mean(dim=2, keepdim=True)
            a_std = best_actions.std(dim=2, unbiased=False, keepdim=True)

            if return_plan_each_iter:
                _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
                best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size).detach()
                plan_each_iter.append(best_plan.data.clone())
                # plan_each_iter.append(a_mu.squeeze(dim=2).data.clone())

        # save result to warmstart later
        # of size self.H, B, 1, self.a_size
        self.prior_mu = a_mu.clone()
        self.prior_std = a_std.clone()
        self.prior_mu[:-1, :, :, :] = a_mu[1:, :, :, :] # shift time one forward
        self.prior_std[:-1, :, :, :] = a_std[1:, :, :, :] # shift time one forward
        self.prior_mu[-1, :, :, :] = a_mu[-1, :, :, :] # keep last time step
        self.prior_std[-1, :, :, :] = 3 * a_std[-1, :, :, :] # but triple its std to explore


        if return_plan_each_iter:
            return plan_each_iter
        if return_plan:
            return a_mu.squeeze(dim=2)
        else:
            _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
            best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size).detach()
            print("Best plan reward:", returns[topk[0]].item())
            return best_plan[0][0]
            # return a_mu.squeeze(dim=2)[0]
