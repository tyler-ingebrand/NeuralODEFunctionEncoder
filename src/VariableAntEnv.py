import os
import time
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import gymnasium as gym
class VariableAntEnv(gym.Env):

    @staticmethod
    def default_dynamics_variable_ranges():
        return {
            "gravity": (-9.8, -9.8)
        }

    def __init__(self, dynamics_variable_ranges:Dict, *env_args, **env_kwargs):
        '''

        :param dynamics_variable_ranges: A dictionary of ranges for the dynamics variables. The following
        keys are optional. The default values are used if not specified. Provide as a tuple of (lower, upper).
        '''
        super().__init__()
        if dynamics_variable_ranges.get("gravity") is not None:
            assert dynamics_variable_ranges["gravity"][0] <= 0 and dynamics_variable_ranges["gravity"][1] <= 0, "gravity must be negative, if not the ant flies away. Negative is down"

        self.env_args = env_args
        self.env_kwargs = env_kwargs
        self.render_mode = env_kwargs.get('render_mode', None)
        self.dynamics_variable_ranges = dynamics_variable_ranges

        # append defaults if not specified
        defaults = self.default_dynamics_variable_ranges()
        for k, v in defaults.items():
            if k not in self.dynamics_variable_ranges:
                self.dynamics_variable_ranges[k] = v

        # placeholder variable
        self.env =  gym.make('Ant-v3', *self.env_args, **self.env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        reset_hps:bool = True,
    ):
        raise Exception("Add current dynamics dict")
        # sample env parameters
        if reset_hps:
            grav = np.random.uniform(*self.dynamics_variable_ranges['gravity'])

            # load env
            # self.env = gym.make('Ant-v3',  *self.env_args, **self.env_kwargs)
            self.env.sim.model.opt.gravity[2] = grav

        # return observation
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


if __name__ == '__main__':
    env = VariableAntEnv({"gravity": (-5,  -1)}, render_mode='human')
    print("dt = ", env.env.unwrapped.dt)
    # loop and render to make sure its working
    for _ in range(10_000):
        if _ % 100 == 0:
            env.reset()
        env.render()
        time.sleep(0.01)
        env.step(env.action_space.sample())
