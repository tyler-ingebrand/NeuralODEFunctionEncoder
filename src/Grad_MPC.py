import numpy as np
import torch
from torch import jit
from torch import nn, optim
import os
import cv2
from tqdm import trange
import colorednoise as cn



class Grad():  # jit.ScriptModule):
    def __init__(self, planning_horizon, opt_iters, samples, top_samples, env, device, action_means, grad_clip=True):
        super().__init__()
        self.set_env(env)
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K = samples
        self.top_K = top_samples
        self.device = device
        self.grad_clip = grad_clip
        self.action_space = self.env.action_space
        self.action_means = 0.07148927728325129 * torch.ones_like(action_means)  # this is a empirically tested hover value
        self.prior_actions = None
        self.prior_action_baseline = None
        self.count = 0

    def set_env(self, env):
        if env.type != "learned":
            raise ValueError("This MPC requires a learned model, so its differentiable.")
        self.env = env
        if self.env is not None:
            self.a_size = env.a_size

    # @jit.script_method
    def forward(self, batch_size, return_plan=False, return_plan_each_iter=False):
        # Here batch is strictly if multiple Plans should be performed!
        B = batch_size

        action_min = self.action_space[:, 0].to(self.device)
        action_max = self.action_space[:, 1].to(self.device)


        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        if self.prior_actions is None:
            action_space_stds = 0.01

            # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
            a_mu = torch.zeros(self.H, B, 1, 2, device=self.device)
            a_std = torch.zeros(self.H, B, 1, 2, device=self.device) + action_space_stds
            n_iters = self.opt_iters
            actions = (a_mu + a_std * torch.randn(self.H, B, self.K, 2, device=self.device)).view(self.H, B * self.K,2)
            actions = actions.clone().detach().requires_grad_(True)
            actions_baseline = torch.ones(self.H, B * self.K, device=self.device) * 0.07148927728325129
            actions_baseline = actions_baseline.clone().detach().requires_grad_(True)

        else:
            n_iters = self.opt_iters
            actions = self.prior_actions.clone().detach()
            actions_baseline = self.prior_action_baseline.clone().detach()
            actions.requires_grad = True
            actions_baseline.requires_grad = True



        # optimizer = optim.SGD([actions], lr=0.1, momentum=0)
        # optimizer = optim.RMSprop([actions], lr=0.1)
        optimizer = optim.Adam([actions, actions_baseline], lr=1e-3)
        plan_each_iter = []

        for index in range(n_iters):
            self.env.reset_state(B*self.K)
            optimizer.zero_grad()

            # calculate actions to apply from the env
            a1 = actions_baseline + actions[:, :, 0]
            a2 = actions_baseline + actions[:, :, 1]
            a3 = actions_baseline - actions[:, :, 0]
            a4 = actions_baseline - actions[:, :, 1]
            actions_to_apply = torch.stack([a1, a2, a3, a4], dim=2)

            # Returns (B*K)
            if index == n_iters - 1:
                returns, predicted_traj_states = self.env.rollout(actions_to_apply, render=False) # self.count >= 10)  # optionally render a plan
            else:
                returns, predicted_traj_states = self.env.rollout(actions_to_apply)
            tot_returns = returns.sum()
            (-tot_returns).backward()

            # grad clip
            # Find norm across batch
            if self.grad_clip:
                epsilon = 1e-6
                max_grad_norm = 1.0
                actions_grad_norm = actions.grad.norm(2.0,dim=2,keepdim=True)+epsilon
                # print("before clip", actions.grad.max().cpu().numpy())

                # Normalize by that
                actions.grad.data.div_(actions_grad_norm)
                actions.grad.data.mul_(actions_grad_norm.clamp(min=0, max=max_grad_norm))
                # print("after clip", actions.grad.max().cpu().numpy())

            optimizer.step()

            # clamp actions without breaking the gradients
            with torch.no_grad():
                a_min = action_min.max()
                a_max = action_max.min()
                actions_baseline.clamp_(a_min, a_max)
                distance_to_action_min = actions_baseline - a_min
                distance_to_action_max = a_max - actions_baseline
                distance = torch.where(distance_to_action_min < distance_to_action_max, distance_to_action_min, distance_to_action_max)
                actions.clamp_(-distance.unsqueeze(2).repeat(1,1,2), distance.unsqueeze(2).repeat(1,1,2))


            _, topk = returns.reshape(B, self.K).topk(self.top_K, dim=1, largest=True, sorted=False)
            # topk += self.K * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
            # best_actions = actions[:, topk.view(-1)].reshape(self.H, B, self.top_K, self.a_size)
            # a_mu = best_actions.mean(dim=2, keepdim=True)
            # a_std = best_actions.std(dim=2, unbiased=False, keepdim=True)

            if return_plan_each_iter:
                _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
                best_plan = actions_to_apply[:, topk[0]].reshape(self.H, B, self.a_size).detach()
                plan_each_iter.append(best_plan.data.clone())

            # There must be cleaner way to do this
            # k_resamp = self.K-self.top_K
            # _, botn_k = returns.reshape(B, self.K).topk(k_resamp, dim=1, largest=False, sorted=False)
            # botn_k += self.K * torch.arange(0, B, dtype=torch.int64, device=self.device).unsqueeze(dim=1)
            #
            # resample_actions = (a_mu + a_std * torch.randn(self.H, B, k_resamp, self.a_size, device=self.device)).view(self.H, B * k_resamp, self.a_size)
            # actions.data[:, botn_k.view(-1)] = resample_actions.data

        actions_to_apply = actions_to_apply.detach()
        # Re-fit belief to the K best action sequences
        _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
        best_plan = actions_to_apply[:, topk[0]].reshape(self.H, B, self.a_size)

        self.prior_actions = actions.detach().clone()
        self.prior_actions[:-1] = self.prior_actions[1:].clone()
        self.prior_action_baseline = actions_baseline.detach().clone()
        self.prior_action_baseline[:-1] = self.prior_action_baseline[1:].clone()


        self.count += 1
        if return_plan_each_iter:
            return plan_each_iter
        if return_plan:
            return best_plan
        else:
            # print(*(f"{t:0.2f}" for t in predicted_traj_states[0, 0, :]))
            return best_plan[0]

