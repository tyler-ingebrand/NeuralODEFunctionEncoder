import warnings

import torch
import math
import copy

from torch import tensor
# from src.Predictors.FE_NeuralODE_Fast import ParallelLinear
try:
    from FE_NeuralODE_Fast import ParallelLinear
except:
    from .FE_NeuralODE_Fast import ParallelLinear
try:
    from Predictor import Predictor
except:
    from .Predictor import Predictor
try:
    from FE_NeuralODE_Residuals import FE_NeuralODE_Residuals
except:
    from .FE_NeuralODE_Residuals import FE_NeuralODE_Residuals

class FE_NeuralODE_Residuals_Fast(Predictor):
    def __init__(self, slow_model:FE_NeuralODE_Residuals):
        super().__init__(slow_model.state_size, slow_model.action_size, slow_model.use_actions)
        self.n_basis = slow_model.n_basis
        self.state_size = slow_model.state_size
        self.action_size = slow_model.action_size
        self.use_actions = slow_model.use_actions
        warnings.warn(
            "This class is fast but memory inefficient. Do not use this for training. Use FE_NeuralODE for training.")

        # copy average function model
        self.model = copy.deepcopy(slow_model.model)

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
                        layers[i].W[j] = torch.nn.Parameter(slow_model.models[j][i].weight.clone())
                        layers[i].b[j] = torch.nn.Parameter(slow_model.models[j][i].bias.clone())
                elif isinstance(slow_model.models[0][i], torch.nn.ReLU):
                    pass
                else:
                    raise Exception("Unknown layer type")
        self.models = torch.nn.Sequential(*layers)

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
        outputs = self.models(inputs.repeat(self.n_basis, 1))
        for basis in range(self.n_basis):
            outputs_slow = slow_model.models[basis](inputs)
            assert torch.allclose(outputs_slow[0], outputs[basis, :], rtol=1e-4, atol=1e-4), f"Model {basis} does not match slow model, got {outputs_slow[0], outputs[basis, :]}"

        # verify output is same for average function
        outputs = self.model(inputs)
        outputs_slow = slow_model.model(inputs)
        assert torch.allclose(outputs_slow, outputs, rtol=1e-4, atol=1e-4), f"Model does not match slow model, got {outputs_slow, outputs}"

        self.representation = None

    def predict_xdot(self, states, actions, representation, use_average=True, use_basis=True):
        assert use_average or use_basis, "At least one of use_average or use_basis must be True"
        inputs = torch.cat([states, actions], dim=-1) if self.use_actions else states
        if use_average and use_basis:
            xdots_average = self.model(inputs)[:, 0:1, :]
            x_dots = self.models(inputs)
            outs = torch.einsum("dkn,kn->dn", x_dots, representation).unsqueeze(1) + xdots_average
            return outs
        elif use_average:
            xdots_average = self.model(inputs)
            return xdots_average
        elif use_basis:
            x_dots = self.models(inputs)
            outs = torch.einsum("dkn,kn->dn", x_dots, representation).unsqueeze(1)
            return outs
    def rk4(self, states, actions, representation, dt=0.05, absolute=True, use_average=True, use_basis=True):
        k1 = self.predict_xdot(states, actions, representation, use_average, use_basis)
        k2 = self.predict_xdot(states + k1 * dt / 2, actions, representation, use_average, use_basis)
        k3 = self.predict_xdot(states + k2 * dt / 2, actions, representation, use_average, use_basis)
        k4 = self.predict_xdot(states + k3 * dt, actions, representation, use_average, use_basis)

        if absolute:
            return states[:, 0:1, :] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        else:
            return (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6



    # predicts the next states given states and actions
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
            average_function_example_dif = self.rk4(example_states, example_actions, None, absolute=False, use_average=True, use_basis=False)
            state_dif = example_next_states - example_states - average_function_example_dif
            for i in range(self.n_basis):
                one_dim_rep = torch.zeros(self.n_basis, self.state_size, device=example_states.device)
                one_dim_rep[i, :] = 1.0
                basis_prediction = self.rk4(example_states.unsqueeze(1).repeat(1, self.n_basis, 1),
                                            example_actions.unsqueeze(1).repeat(1, self.n_basis, 1),
                                            one_dim_rep,
                                            absolute=False,
                                            use_average=False,
                                            use_basis=True)
                basis_prediction = basis_prediction.squeeze(1)
                encodings[i, :] = torch.einsum("dn,dn->n", state_dif, basis_prediction) * (1.0/(basis_prediction.shape[0]))
            self.representation = encodings

    # given an initial state and a list(tensor) of actions, predicts a full trajectory
    def predict_trajectory(self, initial_state: tensor, actions: tensor, example_states: tensor,
                           example_actions: tensor, example_next_states: tensor) -> tensor:
        number_trajectories = initial_state.shape[1]
        time_horizon = actions.shape[2]

        # compute encodings
        representation = self.representation

        # compute traj
        state_predictions = torch.zeros(1, number_trajectories, time_horizon + 1, self.state_size, device=initial_state.device)
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
