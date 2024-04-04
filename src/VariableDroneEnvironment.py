import copy
import os
import time
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
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
        # self.observation_space = self.env.observation_space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)


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
        euler_obs, info = self.env.reset()
        info["euler_obs"] = euler_obs
        obs = self._get_obs()
        info["dynamics"] = self.current_dynamics_dict
        return obs, info

    # gets obs with quarternion instead of euler angles
    def _get_obs(self):
        full_state = self._get_drone_state_vector(0)
        pos, quat, rpy, vel, ang_v, _ = np.split(full_state, [3, 7, 10, 13, 16])
        Rob = np.array(p.getMatrixFromQuaternion(self.env.quat[0])).reshape((3, 3))
        Rbo = Rob.T
        ang_v_body_frame = Rbo @ ang_v
        # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
        obs = np.hstack(
            # [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v]  # Note: world ang_v != body frame pqr
            [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], quat, ang_v_body_frame]
        ).reshape((13,))
        return obs.copy()
    def step(self, action):
        euler_obs, reward, done, info = self.env.step(action)
        info["dynamics"] = self.current_dynamics_dict
        info["euler_obs"] = euler_obs
        terminated = done
        truncated = False
        n_obs = self._get_obs()
        return n_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        '''Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            frame (ndarray): A multidimensional array with the RGB frame captured by PyBullet's camera.
        '''
        res = "hr" # "hr"
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
    controller = "pid" # "Random"


    seed = 2
    np.random.seed(seed)


    # hps = {'M': (0.022, 0.032),
    #         "Ixx": (1.3e-5, 1.5e-5),
    #         "Iyy": (1.3e-5, 1.5e-5),
    #         "Izz": (2.1e-5, 2.2e-5)
    #        }
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
    hps = {'M': (0.02, 0.04 ), # .032),
            "Ixx": (1.4e-5, 1.4e-5),
            "Iyy": (1.4e-5, 1.4e-5),
            "Izz": (2.15e-5, 2.15e-5),
           }
    # mass range: 0.015 to 0.055
    env = VariableDroneEnv(hps, render_mode='human', seed=seed)
    obs, info = env.reset()
    # hover_val = (env.action_space.high[0] + env.action_space.low[0]) * 0.4050312  # this is pretty close to hovering
    hover_val = env.action_space.low[0] + (env.action_space.high[0] - env.action_space.low[0]) * 0.5
    # get dt of sim
    print("dt", 1/env.env.CTRL_FREQ)
    print(hover_val)
    import math

    if render:
        img = env.render()
        width, height = img.shape[1], img.shape[0]
        out = cv2.VideoWriter('../DroneEnvironmentTest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    truncated, terminated = False, False
    # loop and render to make sure its working
    for _ in range(20):
        if _ == 0 or terminated or truncated:
            if terminated:
                print("terminated", _)
            if truncated:
                print("truncated", _)
            obs, info = env.reset(reset_hps=True)
            # env.set_state(      [0, 0, 0,
            #                     0, 1, 0,
            #                     0, 0, 0,
            #                     0, 0, 0])
            obs = env._get_obs()
            # print(obs[6:9])
            hps = info["dynamics"]
            hps = {k: (v, v) for k, v in hps.items()}
            # print(hps)
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
            action = np.array([hover_val, hover_val, hover_val, hover_val])
            # print("z=", obs[4])
        else:
            action = ctrl.select_action(obs, info)
        # action = [0.0705, 0.07, 0.0695, 0.07]

        obs, reward, terminated, truncated, info = env.step(action)
        quat1 = obs[6]
        quat2 = obs[7]
        quat3 = obs[8]
        quat4 = obs[9]

        # normalize quat
        mag = math.sqrt(quat1 ** 2 + quat2 ** 2 + quat3 ** 2 + quat4 ** 2)
        quat1 /= mag
        quat2 /= mag
        quat3 /= mag
        quat4 /= mag

        psi = math.atan2(2 * (quat1 * quat2 + quat3 * quat4), 1 - 2 * (quat2 ** 2 + quat3 ** 2))
        theta = -math.asin(2 * (quat1 * quat3 - quat4 * quat2))
        phi = math.atan2(2 * (quat1 * quat4 + quat2 * quat3), 1 - 2 * (quat3 ** 2 + quat4 ** 2))
        if phi > math.pi/2:
            phi -=  math.pi
        elif phi < -math.pi/2:
            phi += math.pi
        phi = phi * -1
        # print(f"Calculated: {phi:0.2f}, {theta:0.2f}, {psi:0.2f}")
        # print(f"    Actual: {info['euler_obs'][6]:0.2f}, {info['euler_obs'][7]:0.2f}, {info['euler_obs'][8]:0.2f}")
        assert np.allclose([phi, theta, psi], info['euler_obs'][6:9], atol=0.1), "Euler angles do not match"

    if render:
        print("Saving")
        out.release()