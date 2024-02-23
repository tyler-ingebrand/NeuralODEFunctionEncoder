from typing import Tuple

import torch
from torch import tensor

try:
    from Predictor import Predictor
except:
    from .Predictor import Predictor
class Oracle(Predictor):
    def __init__(self, state_size:int, action_size:int,  min_vals_hps, max_vals_hps, use_actions:bool=True, hidden_size=512):
        super().__init__(state_size, action_size, use_actions)
        self.state_size = state_size
        self.action_size = action_size
        self.use_actions = use_actions

        # convert to tensor
        min_vals_hps_tensor = torch.tensor([min_vals_hps[key] for key in min_vals_hps])
        max_vals_hps_tensor = torch.tensor([max_vals_hps[key] for key in max_vals_hps])
        self.min_vals_hps = torch.nn.parameter.Parameter(min_vals_hps_tensor, requires_grad=False) # we will use these to normalize
        self.max_vals_hps = torch.nn.parameter.Parameter(max_vals_hps_tensor, requires_grad=False)

        input_size = state_size + (action_size if use_actions else 0) + len(min_vals_hps)
        output_size = state_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def predict_xdot(self, states, actions, hidden_params:tensor):
        inputs = torch.cat([states, actions, hidden_params], dim=-1) if self.use_actions else torch.cat([states, hidden_params])
        return self.model(inputs)

    def rk4(self, states, actions, hidden_params:tensor, dt=0.05):
        k1 = self.predict_xdot(states, actions, hidden_params)
        k2 = self.predict_xdot(states + k1 * dt / 2, actions, hidden_params)
        k3 = self.predict_xdot(states + k2 * dt / 2, actions, hidden_params)
        k4 = self.predict_xdot(states + k3 * dt, actions, hidden_params)
        return states + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    # predicts the next states given states and actions
    def predict(self, states:tensor, actions:tensor, hidden_params:list[dict],  average_function_only=False) -> Tuple[tensor, dict]:
        assert states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(states.shape[-1], self.state_size)
        assert actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(actions.shape[-1], self.action_size)
        # convert hidden params to tensor from a list of dicts. Also normalize it.
        hidden_params_tensor = torch.tensor([[hidden_params[i][key] for key in hidden_params[i]] for i in range(len(hidden_params))], device=states.device)
        hidden_params_tensor = (hidden_params_tensor - self.min_vals_hps) / (self.max_vals_hps - self.min_vals_hps)
        hidden_params_tensor = hidden_params_tensor.unsqueeze(1).repeat(1, states.shape[1], 1)
        return self.rk4(states, actions, hidden_params_tensor), {}


    # given an initial state and a list(tensor) of actions, predicts a full trajectory
    def predict_trajectory(self, initial_state:tensor, actions:tensor, hidden_params:list[dict]) -> tensor:
        number_envs = initial_state.shape[0]
        number_trajectories = initial_state.shape[1]
        time_horizon = actions.shape[2]
        assert initial_state.shape == (number_envs, number_trajectories, self.state_size)
        assert actions.shape == (number_envs, number_trajectories, time_horizon, self.action_size)

        state_predictions = torch.zeros(number_envs, number_trajectories, time_horizon + 1, self.state_size, device=initial_state.device)
        state_predictions[:, :, 0, :] = initial_state
        for j in range(state_predictions.shape[0]):
            for i in range(time_horizon):
                # predict next state, save it
                current_action = actions[j:j+1, :, i, :]
                current_states = state_predictions[j:j+1, :, i, :]
                next_state_predictions, _ = self.predict(current_states, current_action, [hidden_params[j]])
                state_predictions[j, :, i+1, :] = next_state_predictions

        return state_predictions[:, :, 1:, :]


if __name__ == "__main__":
    model = Oracle(5, 3, True)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = Oracle(5, 3, True)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)


    model = Oracle(5, 3, False)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = NeuralODE(5, 3, False)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)

