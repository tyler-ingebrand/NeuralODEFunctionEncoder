from typing import Tuple

import torch
from torch import tensor
from tqdm import trange

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
        # print(encodings[0])
        # exit(-1)

        # approximate next states
        individual_encodings = torch.zeros(*batch_dims, self.n_basis, self.state_size, device=states.device)
        for i in range(self.n_basis):
            individual_encodings[:, :, i, :] = self.rk4(self.models[i], states, actions, absolute=False)
        return torch.einsum("fks,feks->fes", encodings, individual_encodings) + states, {}


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


    def verify_performance(self, all_states_test, all_actions_test, all_next_states_test, mean, std, device, normalize=True):
        if normalize:
            all_states_test = (all_states_test - mean) / std
            all_next_states_test = (all_next_states_test - mean) / std

        # now do the same thing for testing only
        n_envs_at_once = 10
        n_example_points = 200
        grad_accumulation_steps = 5
        total_test_loss = 0.0
        for grad_accum_step in range(grad_accumulation_steps):
            # randomize
            # get some envs
            env_indicies = torch.randperm(all_states_test.shape[0])[:n_envs_at_once]

            # get some random steps
            perm = torch.randperm(all_states_test.shape[1] * all_states_test.shape[2])
            example_indicies = perm[:n_example_points]
            test_indicies = perm[n_example_points:][:800]  # only gather the first 800 random points
            # print("Env indicies", env_indicies)
            # print("Example indicies", example_indicies)

            # convert to episode and timestep indicies
            example_episode_indicies = example_indicies // all_states_test.shape[2]
            example_timestep_indicies = example_indicies % all_states_test.shape[2]
            test_episode_indicies = test_indicies // all_states_test.shape[2]
            test_timestep_indicies = test_indicies % all_states_test.shape[2]

            # gather data
            states = all_states_test[env_indicies, :, :, :][:, test_episode_indicies, test_timestep_indicies, :].to(device)
            actions = all_actions_test[env_indicies, :, :, :][:, test_episode_indicies, test_timestep_indicies,:].to(device)
            next_states = all_next_states_test[env_indicies, :, :, :][:, test_episode_indicies,test_timestep_indicies, :].to(device)
            example_states = all_states_test[env_indicies, :, :, :][:, example_episode_indicies, example_timestep_indicies, :].to(device)
            example_actions = all_actions_test[env_indicies, :, :, :][:, example_episode_indicies,example_timestep_indicies, :].to(device)
            example_next_states = all_next_states_test[env_indicies, :, :, :][:, example_episode_indicies,example_timestep_indicies, :].to(device)

            # reshape to ignore the episode dim, since we are only doing 1 step, it doesnt matter
            states = states.view(states.shape[0], -1, states.shape[-1])
            actions = actions.view(actions.shape[0], -1, actions.shape[-1])
            next_states = next_states.view(next_states.shape[0], -1, next_states.shape[-1])
            example_states = example_states.view(example_states.shape[0], -1, example_states.shape[-1])
            example_actions = example_actions.view(example_actions.shape[0], -1, example_actions.shape[-1])
            example_next_states = example_next_states.view(example_next_states.shape[0], -1, example_next_states.shape[-1])

            # test the predictor
            predicted_next_states, test_info = self.predict(states, actions, example_states, example_actions, example_next_states)
            test_loss = torch.nn.functional.mse_loss(predicted_next_states, next_states)
            # print(states.shape, actions.shape, next_states.shape, example_states.shape, example_actions.shape, example_next_states.shape, predicted_next_states.shape)
            # print(states[0, 0])
            # print(actions[0, 0])
            # print(next_states[0, 0])
            # print(predicted_next_states[0, 0])
            # exit(-1)
            total_test_loss += test_loss.item() / grad_accumulation_steps
            del states, actions, next_states, example_states, example_actions, example_next_states, predicted_next_states, test_loss
        print(f"FE Neural ODE Test Performance: {total_test_loss:.4f}")

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
