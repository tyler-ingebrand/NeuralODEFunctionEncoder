from typing import Tuple

import torch
from torch import tensor

# the virtual class for predictor models.
class Predictor(torch.nn.Module):
    def __init__(self, state_size:int, action_size:int, use_actions:bool=True):
        super().__init__()


    # predicts the next states given states and actions
    def predict(self, states:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor, average_function_only=False) -> Tuple[tensor, dict]:
        pass

    # given an initial state and a list(tensor) of actions, predicts a full trajectory
    def predict_trajectory(self, initial_state:tensor, actions:tensor, example_states:tensor, example_actions:tensor, example_next_states:tensor) -> tensor:
        pass


