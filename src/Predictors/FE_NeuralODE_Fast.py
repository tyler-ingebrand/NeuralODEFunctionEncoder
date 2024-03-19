import warnings

import torch
import math

from torch import tensor

try:
    from Predictor import Predictor
except:
    from .Predictor import Predictor
try:
    from FE_NeuralODE import FE_NeuralODE
except:
    from .FE_NeuralODE import FE_NeuralODE


class ParallelLinear(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_parallel, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        n = n_inputs
        m = n_outputs
        p = n_parallel
        self.W = torch.nn.Parameter(torch.zeros((p, m, n), **factory_kwargs))
        self.b = torch.nn.Parameter(torch.zeros((p, m), **factory_kwargs))
        self.n = n
        self.m = m
        self.p = p
        # self.reset_parameters()

    def reset_parameters(self) -> None:
        # this function was designed for a single layer, so we need to do it k times to not break their code.
        # this is slow but we only pay this cost once.
        for i in range(self.p):
            torch.nn.init.kaiming_uniform_(self.W[i, :, :], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W[i, :, :])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.b[i, :], -bound, bound)

    def forward(self, x):
        assert x.shape[-1] == self.n, f"Input size of model '{self.n}' does not match input size of data '{x.shape[-1]}'"
        assert x.shape[-2] == self.p, f"Batch size of model '{self.p}' does not match batch size of data '{x.shape[-2]}'"
        y = torch.einsum("pmn,...pn->...pm", self.W, x) + self.b
        return y.squeeze(-1)

    def num_params(self):
        return self.W.numel() + self.b.numel()

    def __repr__(self):
        return f"ParallelLinear({self.n}, {self.m}, {self.p})"

    def __str__(self):
        return self.__repr__()
    def __call__(self, x):
        return self.forward(x)




class FE_NeuralODE_Fast(Predictor):

    def __init__(self, slow_model:FE_NeuralODE):
        super().__init__(slow_model.state_size, slow_model.action_size, slow_model.use_actions)
        self.n_basis = slow_model.n_basis
        self.state_size = slow_model.state_size
        self.action_size = slow_model.action_size
        self.use_actions = slow_model.use_actions
        warnings.warn("This class is fast but memory inefficient. Do not use this for training. Use FE_NeuralODE for training.")

        dtype = torch.float32

        # create memory to store it
        layers = []
        for layer in slow_model.models[0]:
            if isinstance(layer, torch.nn.Linear):
                layers.append(ParallelLinear(layer.in_features, layer.out_features, self.n_basis))
            elif isinstance(layer, torch.nn.ReLU):
                layers.append(torch.nn.ReLU())
            else:
                raise Exception("Unknown layer type")

        # now we need to copy the weights
        with torch.no_grad():
            for i in range(len(layers)):
                if isinstance(slow_model.models[0][i], torch.nn.Linear):
                    for j in range(self.n_basis):
                        layers[i].W[j] =  torch.nn.Parameter(slow_model.models[j][i].weight.clone())
                        layers[i].b[j] =  torch.nn.Parameter(slow_model.models[j][i].bias.clone())
                elif isinstance(slow_model.models[0][i], torch.nn.ReLU):
                    pass
                else:
                    raise Exception("Unknown layer type")
        self.model = torch.nn.Sequential(*layers)

        # now we need to check that the layers are exactly the same
        for i in range(len(layers)):
            if isinstance(slow_model.models[0][i], torch.nn.Linear):
                for j in range(self.n_basis):
                    assert torch.all(slow_model.models[j][i].weight == layers[i].W[j]), f"Layer {i} does not match slow model"
                    assert torch.all(slow_model.models[j][i].bias == layers[i].b[j]), f"Layer {i} does not match slow model"
            elif isinstance(slow_model.models[0][i], torch.nn.ReLU):
                pass
            else:
                raise Exception("Unknown layer type")

        # verify output is the same between models
        inputs = torch.rand((1, self.state_size + (self.action_size if self.use_actions else 0)))
        outputs = self.model(inputs.repeat(self.n_basis, 1))
        for basis in range(self.n_basis):
            outputs_slow = slow_model.models[basis](inputs)
            assert torch.allclose(outputs_slow[0], outputs[basis, :], rtol=1e-4, atol=1e-7), f"Model {basis} does not match slow model"

        self.representation = None


    def predict_xdot(self, states, actions, representation):
        inputs = torch.cat([states, actions], dim=-1) if self.use_actions else states
        xdots = self.model(inputs)
        outs = torch.einsum("dkn,kn->dn", xdots, representation).unsqueeze(1)
        return outs

    def rk4(self, states, actions, representation, dt=0.05, absolute=True):
        k1 = self.predict_xdot(states, actions, representation)
        k2 = self.predict_xdot(states + k1 * dt / 2, actions, representation)
        k3 = self.predict_xdot(states + k2 * dt / 2, actions, representation)
        k4 = self.predict_xdot(states + k3 * dt, actions, representation)

        if absolute:
            return states[:, 0:1, :] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        else:
            return (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    def predict(self, states, actions, representation, average_function_only=False):
        assert states.shape[-1] == self.state_size, f"Input size is {states.shape[-1]}, expected {self.state_size}"
        assert actions.shape[-1] == self.action_size, f"Input size is {actions.shape[-1]}, expected {self.action_size}"

        # approximate next states
        next_states = self.rk4(states, actions, representation)
        return next_states, {}

    def compute_representations(self, example_states, example_actions, example_next_states):
        with torch.no_grad():
            # compute encodings
            example_batch_dims = example_states.shape[:-1]
            encodings = torch.zeros(self.n_basis, self.state_size, device=example_states.device) # batch_dims[:-1] is the batch size minus the number of episodes
            state_dif = example_next_states - example_states
            for i in range(self.n_basis):
                one_dim_rep = torch.zeros(self.n_basis, self.state_size, device=example_states.device)
                one_dim_rep[i, :] = 1.0
                basis_prediction = self.rk4(example_states.unsqueeze(1).repeat(1, self.n_basis, 1),
                                            example_actions.unsqueeze(1).repeat(1, self.n_basis, 1),
                                            one_dim_rep,
                                            absolute=False)
                basis_prediction = basis_prediction.squeeze(1)
                encodings[i, :] = torch.einsum("dn,dn->n", state_dif, basis_prediction) * (1.0/(basis_prediction.shape[0]))
            self.representation = encodings

    def predict_trajectory(self, initial_state:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor) -> tensor:
        number_trajectories = initial_state.shape[1]
        time_horizon = actions.shape[2]

        # compute encodings
        representation = self.representation

        # compute traj
        state_predictions = torch.zeros (1, number_trajectories, time_horizon + 1, self.state_size, device=initial_state.device)
        state_predictions[:, :, 0, :] = initial_state
        for i in range(time_horizon):
            # predict next state, save it
            current_action = actions[0, :, i, :]
            current_states = state_predictions[0, :, i, :]
            next_state_predictions, _ = self.predict(current_states.unsqueeze(1).repeat(1, self.n_basis, 1),
                                                     current_action.unsqueeze(1).repeat(1, self.n_basis, 1),
                                                     representation)
            next_state_predictions = next_state_predictions.squeeze(1)
            state_predictions[0, :, i + 1, :] = next_state_predictions

        return state_predictions[:, :, 1:, :]


    def num_params(self):
        return sum([layer.num_params() for layer in self.model])







