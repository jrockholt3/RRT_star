import Robot_Env
from Robot_Env import dt, t_limit
import numpy as np
from rtree import index
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Robot_Env_v2 import RobotEnv
from search_space import SearchSpace
from optimized_functions import proximity, njit_forward, calc_jnt_err, PDControl, nxt_state

env = RobotEnv()

X = SearchSpace((.6,.6,.9), env)
# print('t=0', X.obs_pos[0])
# print('t=0', X.obs_pos[1])

th = np.array([0, 0, 0], dtype=float)
w = np.array([0.0,0.3,0.0])
th_goal = np.array([np.pi/4, np.pi/4, np.pi/4])
err = calc_jnt_err(th, th_goal)
dedt = -1*w
tau = PDControl(err,dedt)

l = env.robot.links
S = env.robot.S
aph = env.robot.aph

c = np.sqrt(2)/2
curr_pos = np.array([[c, c, .9]])
print('curr_pos shape', curr_pos.shape)

package = nxt_state(curr_pos, th, w, tau, aph, l, S)
print('prox', proximity(curr_pos[:,0], th, aph, l, S))
print('forward', np.round(njit_forward(th, aph, l, S, P_3=np.array([.3,0,0.0,1.0])),2))
print('package', package)