import os
import time

import cv2
import numpy as np
import torch
from tqdm import trange

from gather_trajectories import load_data

from VariableDroneEnvironment import VariableDroneEnv


# script parameters
env_str = 'drone' # "Ant-v3" # "HalfCheetah-v3" # 'drone

xyz = (0, 0, 1)
xyz_vel = (0,0,0)
phy_theta_psi = (0,0,1)
phy_theta_psi_dot = (0,0,0)
state = [xyz[0], xyz_vel[0], xyz[1], xyz_vel[1], xyz[2], xyz_vel[2], phy_theta_psi[0], phy_theta_psi[1], phy_theta_psi[2], phy_theta_psi_dot[0], phy_theta_psi_dot[1], phy_theta_psi_dot[2]]

env = VariableDroneEnv({}, render_mode="rgb_array")

_ = env.reset()
img = env.render()
width, height = img.shape[1], img.shape[0]




env.set_state(state)
true_img = env.render()
true_img = cv2.cvtColor(true_img, cv2.COLOR_RGB2BGR)

# save image
cv2.imwrite("true_img.png", true_img)
