import numpy as np
import torch
import cv2
from tqdm import trange
class MPCEnv:
    def __init__(self, model, state_space, action_space, initial_state, example_states, example_actions, example_next_states, mean, std, normalize=True, render_env=None):
        self.model = model
        if isinstance(model, torch.nn.Module):
            self.type = "learned"
        else:
            self.type = "true"
        self.state_space = state_space
        self.action_space = action_space
        self.state = None
        self.initial_state = initial_state

        # save example data for predictor
        self.example_states = example_states
        self.example_actions = example_actions
        self.example_next_states = example_next_states
        self.normalize = normalize

        # save mean and std for predictor
        self.mean = mean
        self.std = std
        self.counter = 0


        self.render_env = render_env
        # required by interface
        self.a_size = action_space.shape[0]

    def set_state(self, state):
        assert state.shape == self.initial_state.shape
        self.state = state
        self.counter = 0


    # required by interface
    def reset_state(self, batch_size:int)  -> None:
        self.state = torch.tensor(self.initial_state).repeat(batch_size, 1)

    # required by interface
    def rollout(self, actions:torch.tensor, render=False) -> torch.tensor:
        assert len(actions.shape) == 3
        assert actions.shape[-1] == self.a_size
        inital_state = self.state # batch x state_size
        inital_state = inital_state.unsqueeze(0) # 1 x batch x state_size


        if self.type == "learned":
            # reformat example data
            example_states = self.example_states.reshape(1, self.example_states.shape[0], self.example_states.shape[1])
            example_actions = self.example_actions.reshape(1, self.example_actions.shape[0], self.example_actions.shape[1])
            example_next_states = self.example_next_states.reshape(1, self.example_next_states.shape[0], self.example_next_states.shape[1])
            trajectory_states = self.model.predict_trajectory(inital_state, actions.transpose(0, 1).unsqueeze(0), example_states, example_actions, example_next_states)
            # append initial state
            trajectory_states = torch.cat([inital_state.unsqueeze(2), trajectory_states], dim=2)

            # compute some reward here
            # lets use distance of the last state from 0,0,1
            if self.normalize:
                trajectory_states = trajectory_states.squeeze(0) * self.std + self.mean
        else:
            actions = actions.cpu().numpy()
            trajectory_states = np.zeros((actions.shape[1], actions.shape[0], inital_state.shape[
                -1]))  # self.learned_model.predict_trajectory(inital_state, actions, example_states, example_actions, example_next_states)
            for sample_index in range(actions.shape[1]):
                self.model.set_state(inital_state[0, sample_index, :])
                for time_index in range(actions.shape[0]):
                    action = actions[time_index, sample_index, :]
                    obs, reward, _, _, info = self.model.step(action)
                    trajectory_states[sample_index, time_index, :] = obs

            trajectory_states = torch.tensor(trajectory_states)
            actions = torch.tensor(actions)

        # render a trajectory to see whats happening
        if render:
            with torch.no_grad():
                print("Rendering to hallucinated_mpc.mp4 and real_mpc.mp4")
                traj_to_render = 0
                img = self.render_env.render()
                width, height = img.shape[1], img.shape[0]
                out = cv2.VideoWriter(f'hallicunated_mpc.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
                for time_index in range(trajectory_states.shape[1]):
                    self.render_env.set_state(trajectory_states[traj_to_render, time_index, :].detach().cpu().numpy())
                    img = self.render_env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out.write(img)
                out.release()
                del out

                # now actally apply those actions
                out = cv2.VideoWriter(f'real_mpc.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
                self.render_env.set_state(trajectory_states[traj_to_render, 0, :].detach().cpu().numpy())
                for time_index in range(trajectory_states.shape[1]):
                    img = self.render_env.render()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out.write(img)
                    self.render_env.step(actions[traj_to_render, time_index, :].detach().cpu().numpy())
                out.release()
                del out
                print("Continuing")
                err # this must be here


        # hover action
        hover_action = 0.07148927728325129  # this is emprically tested
        hover_actions = torch.tensor([hover_action, hover_action, hover_action, hover_action],
                                     device=trajectory_states.device)
        close_to_hover_reward = -torch.norm(actions - hover_actions, dim=2).mean(dim=0)

        # change in action
        action_change = torch.norm(actions[:, 1:] - actions[:, :-1], dim=2).mean(dim=0)
        action_change = torch.cat([torch.zeros(1 , device=action_change.device), action_change])
        action_reward = -action_change

        # distance from last state to 0,0,1
        x = trajectory_states[:, :, 0]
        y = trajectory_states[:, :, 2]
        z = trajectory_states[:, :, 4]
        distance = torch.sqrt(x ** 2 + y ** 2 + (z - 1) ** 2)
        reward_distance = -distance.mean(dim=1)

        # keep it stable every step so it doesnt turb over
        phi = trajectory_states[:, :, 6]
        theta = trajectory_states[:, :, 7]
        psi = trajectory_states[:, :, 8]
        stability = phi ** 2 + theta ** 2 + psi ** 2
        reward_stability = -stability.mean(dim=1)

        # low velocity
        x_dot = trajectory_states[:, :, 1]
        y_dot = trajectory_states[:, :, 3]
        z_dot = trajectory_states[:, :, 5]
        velocity = x_dot ** 2 + y_dot ** 2 + 5 * z_dot ** 2
        reward_velocity = -velocity.mean(dim=1)

        # low rotational velocity
        # state labels: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
        #                                'init_phi', 'init_theta', 'init_psi', 'init_p', 'init_q', 'init_r']
        p = trajectory_states[:, :, 9]
        q = trajectory_states[:, :, 10]
        r = trajectory_states[:, :, 11]
        rotational_velocity = p ** 2 + q ** 2 + r ** 2
        reward_rotational_velocity = -rotational_velocity.mean(dim=1)

        # reward alive
        alive = z > 0.1
        reward_alive = alive.float()
        reward_alive = reward_alive.mean(dim=1)

        # tune this
        weight_distance = 5
        weight_stability = 25
        weight_velocity = 0
        weight_rotational_velocity = 0
        weight_alive = 0
        weight_hover = 15
        weight_action_change = 15

        scales = [weight_distance * reward_distance,
                  weight_stability * reward_stability,
                  weight_velocity * reward_velocity,
                  weight_rotational_velocity * reward_rotational_velocity,
                  weight_alive * reward_alive,
                  weight_hover * close_to_hover_reward,
                  weight_action_change * action_reward,

                  ]
        scales_means = [s.mean().item() for s in scales]
        # print(f"{self.counter}:", *(f'{scales_means[i]:.2f} ' for i in range(len(scales_means))))
        traj_return = torch.zeros_like(scales[0])
        for scale in scales:
            traj_return += scale
        self.counter += 1
        return traj_return
