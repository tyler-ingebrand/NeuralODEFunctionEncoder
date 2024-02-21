from typing import Tuple

import torch
from torch import tensor
from tqdm import trange

try:
    from Predictor import Predictor
except:
    from .Predictor import Predictor
class FE_Residuals(Predictor):
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
        self.average_model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.state_size)
        )


    # predicts the next states given states and actions
    def predict(self, states:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor, average_function_only=False) -> Tuple[tensor, dict]:
        assert states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(states.shape[-1], self.state_size)
        assert actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(actions.shape[-1], self.action_size)
        assert example_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_states.shape[-1], self.state_size)
        assert example_actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(example_actions.shape[-1], self.action_size)
        assert example_next_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_next_states.shape[-1], self.state_size)

        if not average_function_only:
            # compute average functions
            with torch.no_grad():
                example_inputs = torch.cat([example_states, example_actions], dim=-1) if self.use_actions else example_states
                inputs = torch.cat([states, actions], dim=-1) if self.use_actions else states
                example_average_function = self.average_model(example_inputs)
                average_function = self.average_model(inputs)
            batch_dims = states.shape[:-1]
            example_batch_dims = example_states.shape[:-1]

            # compute encodings from example data
            example_individual_encodings = self.model(example_inputs).view(*example_batch_dims, self.n_basis, self.state_size)
            encodings = torch.einsum("fes,feks->fks", example_next_states - example_average_function, example_individual_encodings) * (1.0/(example_individual_encodings.shape[-3]))

            # approximate next states
            individual_encodings = self.model(inputs).view(*batch_dims, self.n_basis, self.state_size)
            return torch.einsum("fks,feks->fes", encodings, individual_encodings) + average_function, {}
        else:
            inputs = torch.cat([example_states, example_actions], dim=-1) if self.use_actions else example_states
            return self.average_model(inputs), {}


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
        state_predictions[:, :, 0, :] = initial_state
        num_functions_at_once = 40
        for j in range(0, state_predictions.shape[0], num_functions_at_once):
            for i in range(time_horizon):
                # predict next state, save it
                current_action = actions[j:j+num_functions_at_once, :, i, :]
                current_states = state_predictions[j:j+num_functions_at_once, :, i, :]
                next_state_predictions, _ = self.predict(current_states, current_action, example_states[j:j+num_functions_at_once], example_actions[j:j+num_functions_at_once], example_next_states[j:j+num_functions_at_once])
                state_predictions[j:j+num_functions_at_once, :, i+1, :] = next_state_predictions

        return state_predictions[:, :, 1:, :]
if __name__ == "__main__":
    model = FE_Residuals(5, 3, True)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = FE_Residuals(5, 3, True)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)


    model = FE_Residuals(5, 3, False)
    states, actions, example_states, example_actions, example_next_states = torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5), torch.rand(7, 10, 3), torch.rand(7, 10, 5)
    print(torch.sum(torch.tensor([p.numel() for p in model.parameters()])).item(), "parameters")
    out1 = model.predict(states, actions, example_states, example_actions, example_next_states)
    torch.save(model.state_dict(), "/tmp/model.pt")
    del model
    model = FE_Residuals(5, 3, False)
    model.load_state_dict(torch.load("/tmp/model.pt"))
    out2 = model.predict(states, actions, example_states, example_actions, example_next_states)
    assert torch.allclose(out1, out2)
