import torch
from torch import tensor

from Predictor import Predictor

class FE(Predictor):
    def __init__(self, state_size:int, action_size:int, use_actions:bool=True, hidden_size=512, n_basis=100):
        super().__init__(state_size, action_size, use_actions)
        self.state_size = state_size
        self.action_size = action_size
        self.use_actions = use_actions
        self.n_basis = n_basis
        input_size = state_size + (action_size if use_actions else 0)
        output_size = state_size * n_basis
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


    # predicts the next states given states and actions
    def predict(self, states:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor) -> tensor:
        assert states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(states.shape[-1], self.state_size)
        assert actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(actions.shape[-1], self.action_size)
        assert example_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_states.shape[-1], self.state_size)
        assert example_actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(example_actions.shape[-1], self.action_size)
        assert example_next_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_next_states.shape[-1], self.state_size)
        batch_dims = states.shape[:-1]

        # compute encodings from example data
        example_inputs = torch.cat([example_states, example_actions], dim=-1) if self.use_actions else example_states
        example_individual_encodings = self.model(example_inputs).view(*batch_dims, self.n_basis, self.state_size)
        encodings = torch.einsum("...es,...eks->...ks", example_next_states, example_individual_encodings)

        # approximate next states
        inputs = torch.cat([states, actions], dim=-1) if self.use_actions else states
        individual_encodings = self.model(inputs).view(*batch_dims, self.n_basis, self.state_size)
        return torch.einsum("...ks,...eks->...es", encodings, individual_encodings)


    # given an initial state and a list(tensor) of actions, predicts a full trajectory
    def predict_trajectory(self, initial_state:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor) -> tensor:
        raise Exception("Not implemented")

if __name__ == "__main__":
    model = FE(5, 3, True)
    states, actions, example_states, example_actions, example_next_states = torch.rand(10, 5), torch.rand(10, 3), torch.rand(10, 5), torch.rand(10, 3), torch.rand(10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = FE(5, 3, True)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)
    print("FE.py has been tested successfully")