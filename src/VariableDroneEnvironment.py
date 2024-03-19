import copy
import os
import time
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import gymnasium as gym
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType
import pybullet as p
from ExploratoryPID import ExploratoryPID

class VariableDroneEnv(gym.Env):

    @staticmethod
    def default_dynamics_variable_ranges():
        return {
            'M': (0.027, 0.027),
            'Ixx': (1.4e-5, 1.4e-5),
            'Iyy': (1.4e-5, 1.4e-5),
            'Izz': (2.17e-5, 2.17e-5)
        }

    def __init__(self, dynamics_variable_ranges:Dict, *env_args, **env_kwargs):
        '''

        :param dynamics_variable_ranges: A dictionary of ranges for the dynamics variables. The following
        keys are optional. The default values are used if not specified. Provide as a tuple of (lower, upper).
        - M
        - Ixx
        - Iyy
        - Izz
        '''
        super().__init__()
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
        inertial_prop = [0.027, 1.4e-5, 1.4e-5, 2.17e-5] # default
        self.task = 'stabilization'
        if self.task == "traj_tracking":
            self.task_info = {
            'trajectory_type': 'figure8',
            'num_cycles': 1,
            'trajectory_plane': 'xz',
            'trajectory_position_offset': [0.5, 0],
            'trajectory_scale': 0.75,
            'proj_point': [0, 0, 0.5],
            'proj_normal': [0, 1, 1],
            }
        elif self.task == 'stabilization':
            self.task_info = {
                'stabilization_goal': [0, 0, 1],
                'stabilization_goal_tolerance': -0.1, # never terminated based on stabilization goal
            }
        self.env =  Quadrotor(task_info=self.task_info,
                              task=self.task,
                              inertial_prop=inertial_prop,
                              info_in_reset=True,
                              quad_type=QuadType.THREE_D,
                              episode_len_sec=5,
                              ctrl_freq=25,
                              *self.env_args, **self.env_kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.current_dynamics_dict = {}
        self.goal_range = [[-1, 1], [-1, 1], [0.4, 1.6]] # x, y, z

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        reset_hps: bool = True,
        ):
        if reset_hps:
            # sample env parameters
            M = np.random.uniform(*self.dynamics_variable_ranges['M'])
            Ixx = np.random.uniform(*self.dynamics_variable_ranges['Ixx'])
            Iyy = np.random.uniform(*self.dynamics_variable_ranges['Iyy'])
            Izz = np.random.uniform(*self.dynamics_variable_ranges['Izz'])

            current_dynamics_dict = {
                'M': M,
                'Ixx': Ixx,
                'Iyy': Iyy,
                'Izz': Izz
            }
            self.current_dynamics_dict = current_dynamics_dict

        # samples a random goal with wahtever dynamics
        newgoal = [np.random.uniform(*self.goal_range[0]), np.random.uniform(*self.goal_range[1]), np.random.uniform(*self.goal_range[2])]
        self.task_info['stabilization_goal'] = newgoal

        # load env with this xml file
        inertial_prop = self.current_dynamics_dict['M'], self.current_dynamics_dict['Ixx'], self.current_dynamics_dict['Iyy'], self.current_dynamics_dict['Izz']
        self.env.close() # THIS IS CRUCIAL DONT DELETE THIS. OTHERWISE MEMORY LEAK
        del self.env
        self.env = Quadrotor(task_info=self.task_info,
                             task=self.task,
                             inertial_prop=inertial_prop,
                             quad_type=QuadType.THREE_D,
                             info_in_reset=True,
                             done_on_out_of_bound=True,
                             episode_len_sec=5,
                             ctrl_freq=25,
                             *self.env_args, **self.env_kwargs)

        # return observation
        state, info = self.env.reset()
        info["dynamics"] = self.current_dynamics_dict
        return state, info

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        info["dynamics"] = self.current_dynamics_dict
        terminated = done
        truncated = False
        return next_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        '''Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            frame (ndarray): A multidimensional array with the RGB frame captured by PyBullet's camera.
        '''
        res = "low" # "hr"
        if res == "low":
            width, height = 720, 480
        elif res == "medium":
            width, height = 1280, 720
        else:
            width, height = 1920, 1080


        [w, h, rgb, _, _] = p.getCameraImage(width=width,
                                             height=height,
                                             shadow=1,
                                             viewMatrix=self.CAM_VIEW,
                                             projectionMatrix=self.CAM_PRO,
                                             renderer=p.ER_TINY_RENDERER,
                                             flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                             physicsClientId=self.PYB_CLIENT)
        # Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA').show()
        return np.reshape(rgb, (h, w, 4))

    def close(self):
        return self.env.close()

    # pass through any other methods
    def __getattr__(self, name):
        return getattr(self.env, name)

    def set_state(self, state):
        init_values = {'init_x': state[0],
                       'init_y': state[2],
                       'init_z': state[4],
                       'init_x_dot': state[1],
                       'init_y_dot': state[3],
                       'init_z_dot': state[5],
                       'init_phi': state[6],
                       'init_theta': state[7],
                       'init_psi': state[8],
                       'init_p': state[9],
                       'init_q': state[10],
                       'init_r': state[11]
                          }

        INIT_XYZ = [init_values.get('init_' + k, 0.) for k in ['x', 'y', 'z']]
        INIT_VEL = [init_values.get('init_' + k + '_dot', 0.) for k in ['x', 'y', 'z']]
        INIT_RPY = [init_values.get('init_' + k, 0.) for k in ['phi', 'theta', 'psi']]
        INIT_ANG_VEL = [init_values.get('init_' + k, 0.) for k in ['p', 'q', 'r']]


        p.resetBasePositionAndOrientation(self.env.DRONE_IDS[0], INIT_XYZ,
                                          p.getQuaternionFromEuler(INIT_RPY),
                                          physicsClientId=self.env.PYB_CLIENT)
        p.resetBaseVelocity(self.env.DRONE_IDS[0], INIT_VEL, INIT_ANG_VEL,
                            physicsClientId=self.env.PYB_CLIENT)

        # Update BaseAviary internal variables before calling self._get_observation().
        self._update_and_store_kinematic_information()


if __name__ == '__main__':
    from tqdm import trange
    import copy
    import cv2

    render = True
    controller = "hover" # "Random"




    hps = {'M': (0.022, 0.032),
            "Ixx": (1.3e-5, 1.5e-5),
            "Iyy": (1.3e-5, 1.5e-5),
            "Izz": (2.1e-5, 2.2e-5)
           }
    # hps = {'M': (0.032, 0.032),
    #         "Ixx": (1.5e-5, 1.5e-5),
    #         "Iyy": (1.5e-5, 1.5e-5),
    #         "Izz": (2.2e-5, 2.2e-5)
    #        }
    # hps = {'M': (0.027, 0.027),
    #         "Ixx": (1.4e-5, 1.4e-5),
    #         "Iyy": (1.4e-5, 1.4e-5),
    #         "Izz": (2.17e-5, 2.17e-5)
    #        }
    env = VariableDroneEnv(hps, render_mode='human')
    obs, info = env.reset()
    # get dt of sim
    print("dt", 1/env.env.CTRL_FREQ)

    if render:
        img = env.render()
        width, height = img.shape[1], img.shape[0]
        out = cv2.VideoWriter('../DroneEnvironmentTest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    truncated, terminated = False, False
    # loop and render to make sure its working
    for _ in trange(100):
        if _ == 0 or terminated or truncated:
            if terminated:
                print("terminated", _)
            if truncated:
                print("truncated", _)
            obs, info = env.reset(reset_hps=False)
            env.set_state(      [0, 0, 0,
                                0, 1, 0,
                                0, 0, 0,
                                0,0, 0])
            hps = info["dynamics"]
            hps = {k: (v, v) for k, v in hps.items()}
            print(hps)
            env_func = lambda : VariableDroneEnv(hps, render_mode='human')
            ctrl = ExploratoryPID(env_func=env_func)

        # render
        if render:
            img = env.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            out.write(img)

        # step
        if controller == "Random":
            action = env.action_space.sample()
        elif controller == "hover":
            hover_val = (env.action_space.high[0]  + env.action_space.low[0]) * 0.4050312 # this is pretty close to hovering
            print(hover_val)
            action = np.array([hover_val, hover_val, hover_val, hover_val])
            # print("z=", obs[4])
        else:
            action = ctrl.select_action(obs, info)

        obs, reward, terminated, truncated, info = env.step(action)

    if render:
        print("Saving")
        out.release()