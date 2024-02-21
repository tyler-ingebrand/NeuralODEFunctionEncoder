from typing import Tuple

import torch
from torch import tensor
try:
    from Predictor import Predictor
except:
    from .Predictor import Predictor
class Zeros(Predictor):
    def __init__(self, state_size:int, action_size:int, use_actions:bool=True, hidden_size=512):
        super().__init__(state_size, action_size, use_actions)
        self.state_size = state_size
        self.action_size = action_size
        self.use_actions = use_actions

    # predicts the next states given states and actions
    def predict(self, states:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor, average_function_only=False) -> Tuple[tensor, dict]:
        assert states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(states.shape[-1], self.state_size)
        assert actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(actions.shape[-1], self.action_size)
        assert example_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_states.shape[-1], self.state_size)
        assert example_actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(example_actions.shape[-1], self.action_size)
        assert example_next_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_next_states.shape[-1], self.state_size)
        return torch.zeros(states.shape), {}


    # given an initial state and a list(tensor) of actions, predicts a full trajectory
    def predict_trajectory(self, initial_state:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor) -> tensor:
        number_envs = initial_state.shape[0]
        number_trajectories = initial_state.shape[1]
        time_horizon = actions.shape[2]
        number_examples = example_states.shape[1]
        assert initial_state.shape == (number_envs, number_trajectories, self.state_size)
        assert actions.shape == (number_envs, number_trajectories, time_horizon, self.action_size)
        assert example_states.shape == (number_envs, number_examples, self.state_size)
        assert example_actions.shape == (number_envs, number_examples, self.action_size)
        assert example_next_states.shape == (number_envs, number_examples, self.state_size)

        state_predictions = torch.zeros(number_envs, number_trajectories, time_horizon + 1, self.state_size, device=initial_state.device)
        return state_predictions[:, :, 1:, :]

if __name__ == "__main__":
    model = MLP(5, 3, True)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = MLP(5, 3, True)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)

    model = MLP(5, 3, False)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = MLP(5, 3, False)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)