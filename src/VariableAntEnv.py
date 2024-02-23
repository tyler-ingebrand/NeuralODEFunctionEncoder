
import os
import time
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import gymnasium as gym

LEG_LENGTH = (0.2 **2 + 0.2**2)**0.5
ANKLE_LENGTH = (0.4 **2 + 0.4**2)**0.5
class VariableAntEnv(gym.Env):


    @staticmethod
    def default_dynamics_variable_ranges():
        leg_lengths = (0.2 **2 + 0.2**2)**0.5
        ankle_lengths = (0.4 **2 + 0.4**2)**0.5
        gear = 150
        return {
            "gravity": (-9.8, -9.8),
            "front_left_leg_length": (leg_lengths, leg_lengths),
            "front_left_foot_length":(ankle_lengths, ankle_lengths),
            "front_right_leg_length":(leg_lengths, leg_lengths),
            "front_right_foot_length":(ankle_lengths, ankle_lengths),
            "back_left_leg_length":(leg_lengths, leg_lengths),
            "back_left_foot_length":(ankle_lengths, ankle_lengths),
            "back_right_leg_length":(leg_lengths, leg_lengths),
            "back_right_foot_length":(ankle_lengths, ankle_lengths),
            "front_left_gear":(gear, gear),
            "front_right_gear":(gear, gear),
            "back_left_gear":(gear, gear),
            "back_right_gear":(gear, gear),
            "front_left_ankle_gear":(gear, gear),
            "front_right_ankle_gear":(gear, gear),
            "back_left_ankle_gear":(gear, gear),
            "back_right_ankle_gear":(gear, gear),
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

        # path to write xml to
        current_date_time = datetime.now().strftime("%Y%m%d%H%M%S")
        self.xml_path = '/tmp/ants'
        self.xml_name =  f'{current_date_time}.xml'
        os.makedirs(self.xml_path, exist_ok=True)

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        reset_hps:bool = True,
    ):
        # sample env parameters
        if reset_hps:
            grav = np.random.uniform(*self.dynamics_variable_ranges['gravity'])

            front_left_leg_length = np.random.uniform(*self.dynamics_variable_ranges['front_left_leg_length'])
            front_left_foot_length = np.random.uniform(*self.dynamics_variable_ranges['front_left_foot_length'])
            front_right_leg_length = np.random.uniform(*self.dynamics_variable_ranges['front_right_leg_length'])
            front_right_foot_length = np.random.uniform(*self.dynamics_variable_ranges['front_right_foot_length'])
            back_left_leg_length = np.random.uniform(*self.dynamics_variable_ranges['back_left_leg_length'])
            back_left_foot_length = np.random.uniform(*self.dynamics_variable_ranges['back_left_foot_length'])
            back_right_leg_length = np.random.uniform(*self.dynamics_variable_ranges['back_right_leg_length'])
            back_right_foot_length = np.random.uniform(*self.dynamics_variable_ranges['back_right_foot_length'])

            front_left_gear = np.random.uniform(*self.dynamics_variable_ranges['front_left_gear'])
            front_right_gear = np.random.uniform(*self.dynamics_variable_ranges['front_right_gear'])
            back_left_gear = np.random.uniform(*self.dynamics_variable_ranges['back_left_gear'])
            back_right_gear = np.random.uniform(*self.dynamics_variable_ranges['back_right_gear'])
            front_left_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['front_left_ankle_gear'])
            front_right_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['front_right_ankle_gear'])
            back_left_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['back_left_ankle_gear'])
            back_right_ankle_gear = np.random.uniform(*self.dynamics_variable_ranges['back_right_ankle_gear'])

            # load env
            current_dynamics_dict = {
                'gravity': grav,
                'front_left_leg_length': front_left_leg_length,
                'front_left_foot_length': front_left_foot_length,
                'front_right_leg_length': front_right_leg_length,
                'front_right_foot_length': front_right_foot_length,
                'back_left_leg_length': back_left_leg_length,
                'back_left_foot_length': back_left_foot_length,
                'back_right_leg_length': back_right_leg_length,
                'back_right_foot_length': back_right_foot_length,
                'front_left_gear': front_left_gear,
                'front_right_gear': front_right_gear,
                'back_left_gear': back_left_gear,
                'back_right_gear': back_right_gear,
                'front_left_ankle_gear': front_left_ankle_gear,
                'front_right_ankle_gear': front_right_ankle_gear,
                'back_left_ankle_gear': back_left_ankle_gear,
                'back_right_ankle_gear': back_right_ankle_gear,
            }

            # create xml file for these parameters
            path = self.create_xml_file(front_left_leg_length, front_left_foot_length, front_right_leg_length, front_right_foot_length, back_left_leg_length, back_left_foot_length, back_right_leg_length, back_right_foot_length, front_left_gear, front_right_gear, back_left_gear, back_right_gear, front_left_ankle_gear, front_right_ankle_gear, back_left_ankle_gear, back_right_ankle_gear,)
            self.current_dynamics_dict = current_dynamics_dict
            self.env = gym.make('Ant-v3', xml_file=path, *self.env_args, **self.env_kwargs)
            self.env.sim.model.opt.gravity[2] = grav

        # return observation
        state, info = self.env.reset()
        info["dynamics"] = self.current_dynamics_dict
        return state, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        info["dynamics"] = self.current_dynamics_dict
        return next_state, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def create_xml_file(self,
                        front_left_leg_length, front_left_foot_length,
                        front_right_leg_length, front_right_foot_length,
                        back_left_leg_length, back_left_foot_length,
                        back_right_leg_length, back_right_foot_length,
                        front_left_gear, front_right_gear,
                        back_left_gear, back_right_gear,
                        front_left_ankle_gear, front_right_ankle_gear,
                        back_left_ankle_gear, back_right_ankle_gear,
                        ):
        file_string = f"""<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0.0" name="aux_1_geom" size="0.08" type="capsule"/>
        <body name="aux_1" pos="{0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0.0" name="left_leg_geom" size="0.08" type="capsule"/>
          <body pos="{0.2 * front_left_leg_length/LEG_LENGTH}  {0.2 * front_left_leg_length/LEG_LENGTH} 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {0.4 * front_left_foot_length / ANKLE_LENGTH} {0.4 * front_left_foot_length / ANKLE_LENGTH} 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0.0" name="aux_2_geom" size="0.08" type="capsule"/>
        <body name="aux_2" pos="-{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0.0" name="right_leg_geom" size="0.08" type="capsule"/>
          <body pos="-{0.2 * front_right_leg_length/LEG_LENGTH} {0.2 * front_right_leg_length/LEG_LENGTH} 0">
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{0.4 * front_right_foot_length / ANKLE_LENGTH} {0.4 * front_right_foot_length / ANKLE_LENGTH} 0.0" name="right_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0.0" name="aux_3_geom" size="0.08" type="capsule"/>
        <body name="aux_3" pos="-{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0.0" name="back_leg_geom" size="0.08" type="capsule"/>
          <body pos="-{0.2 * back_left_leg_length/LEG_LENGTH} -{0.2 * back_left_leg_length/LEG_LENGTH} 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-70 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{0.4 * back_left_foot_length / ANKLE_LENGTH} -{0.4 * back_left_foot_length / ANKLE_LENGTH} 0.0" name="third_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0.0" name="aux_4_geom" size="0.08" type="capsule"/>
        <body name="aux_4" pos="{0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0.0" name="rightback_leg_geom" size="0.08" type="capsule"/>
          <body pos="{0.2 * back_right_leg_length/LEG_LENGTH} -{0.2 * back_right_leg_length/LEG_LENGTH} 0">
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 70" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {0.4 * back_right_foot_length / ANKLE_LENGTH} -{0.4 * back_right_foot_length / ANKLE_LENGTH} 0.0" name="fourth_ankle_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="{back_right_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="{back_right_ankle_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="{front_left_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="{front_left_ankle_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="{front_right_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="{front_right_ankle_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="{back_left_gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="{back_left_ankle_gear}"/>
  </actuator>
</mujoco>
"""
        with open(os.path.join(self.xml_path, self.xml_name), 'w') as f:
            f.write(file_string)
        return os.path.join(self.xml_path, self.xml_name)











if __name__ == '__main__':
    foot_max, foot_min = ANKLE_LENGTH/2, ANKLE_LENGTH * 1.5
    leg_max, leg_min = LEG_LENGTH/2, LEG_LENGTH * 1.5
    gear_min, gear_max = 100, 200
    hps = { "gravity": (-9.8,  -5),
           "front_left_leg_length": (leg_min, leg_max),
            "front_left_foot_length": (foot_min, foot_max),
            "front_right_leg_length": (leg_min, leg_max),
            "front_right_foot_length": (foot_min, foot_max),
            "back_left_leg_length": (leg_min, leg_max),
            "back_left_foot_length": (foot_min, foot_max),
            "back_right_leg_length": (leg_min, leg_max),
            "back_right_foot_length": (foot_min, foot_max),

            # "front_left_gear": (gear_min, gear_max),
            # "front_right_gear": (gear_min, gear_max),
            # "back_left_gear": (gear_min, gear_max),
            # "back_right_gear": (gear_min, gear_max),
            # "front_left_ankle_gear": (gear_min, gear_max),
            # "front_right_ankle_gear": (gear_min, gear_max),
            # "back_left_ankle_gear": (gear_min, gear_max),
            # "back_right_ankle_gear": (gear_min, gear_max),


    }
    env = VariableAntEnv(hps, render_mode='human')
    print("dt = ", env.env.unwrapped.dt)
    # loop and render to make sure its working
    for _ in range(10_000):
        if _ % 100 == 0:
            env.reset()
        env.render()
        time.sleep(0.01)
        env.step(env.action_space.sample())
