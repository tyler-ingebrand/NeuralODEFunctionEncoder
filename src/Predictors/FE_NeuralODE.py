from typing import Tuple

import torch
from torch import tensor

try:
    from Predictor import Predictor
except:
    from .Predictor import Predictor
class FE_NeuralODE(Predictor):
    def __init__(self, state_size:int, action_size:int, use_actions:bool=True, hidden_size=51, n_basis=100):
        super().__init__(state_size, action_size, use_actions)
        self.state_size = state_size
        self.action_size = action_size
        self.use_actions = use_actions
        self.n_basis = n_basis
        input_size = state_size + (action_size if use_actions else 0)
        output_size = state_size
        self.models = torch.nn.ParameterList([torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        ) for _ in range(n_basis)])

    def predict_xdot(self, model, states, actions):
        inputs = torch.cat([states, actions], dim=-1) if self.use_actions else states
        return model(inputs)

    def rk4(self, model, states, actions, dt=0.05, absolute=True):
        k1 = self.predict_xdot(model, states, actions)
        k2 = self.predict_xdot(model, states + k1 * dt / 2, actions)
        k3 = self.predict_xdot(model, states + k2 * dt / 2, actions)
        k4 = self.predict_xdot(model, states + k3 * dt, actions)
        if absolute:
            return states + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        else:
            return (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


    # predicts the next states given states and actions
    def predict(self, states:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor, average_function_only=False) -> Tuple[tensor, dict]:
        assert states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(states.shape[-1], self.state_size)
        assert actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(actions.shape[-1], self.action_size)
        assert example_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_states.shape[-1], self.state_size)
        assert example_actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(example_actions.shape[-1], self.action_size)
        assert example_next_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_next_states.shape[-1], self.state_size)

        # compute encodings
        batch_dims = states.shape[:-1]
        example_batch_dims = example_states.shape[:-1]
        encodings = torch.zeros(*example_batch_dims[:-1], self.n_basis, self.state_size, device=states.device) # batch_dims[:-1] is the batch size minus the number of episodes
        for i in range(self.n_basis):
            state_dif = example_next_states - example_states
            basis_prediction = self.rk4(self.models[i], example_states, example_actions, absolute=False)
            encodings[:, i, :] = torch.einsum("fes,fes->fs", state_dif, basis_prediction) * (1.0/(basis_prediction.shape[-2]))
        encodings = encodings

        # approximate next states
        individual_encodings = torch.zeros(*batch_dims, self.n_basis, self.state_size, device=states.device)
        for i in range(self.n_basis):
            individual_encodings[:, :, i, :] = self.rk4(self.models[i], states, actions, absolute=False)
        return torch.einsum("fks,feks->fes", encodings, individual_encodings) + states, {}


    # given an initial state and a list(tensor) of actions, predicts a full trajectory
    def predict_trajectory(self, initial_state:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor) -> tensor:
        raise Exception("Not implemented")

if __name__ == "__main__":
    model = FE_NeuralODE(5, 3, True)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = FE_NeuralODE(5, 3, True)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)

    model = FE_NeuralODE(5, 3, False)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = FE_NeuralODE(5, 3, False)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)
