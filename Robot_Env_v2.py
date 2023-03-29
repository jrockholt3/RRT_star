import numpy as np
from numba import jit
from env_config import dt, t_limit, thres, vel_thres, prox_thres, min_prox, vel_prox, tau_max, damping, P, D, jnt_vel_max, j_max
from optimized_functions import nxt_state, calc_jnt_err, PDControl, angle_calc
from support_classes import vertex
from Robot3D import robot_3link
from Object_v2 import rand_object

robot = robot_3link()
a = robot.aph
l = robot.links
S = robot.S
rng = np.random.default_rng()

def env_replay(th, w, t_start, th_goal, obs_dict, steps):
    t = t_start
    jnt_err = calc_jnt_err(th, th_goal)
    dedt  = w * np.sign(jnt_err) 

    if t>= t_limit/dt:
        reward = -np.inf
        return th, w, reward, False

    score = 0
    done = False
    while not done and t<t_start+steps and t < t_limit/dt:
        tau = PDControl(jnt_err, dedt)
        obj_arr = obs_dict[t]
        temp = nxt_state(obj_arr, th, w, tau, a, l, S)
        nxt_th = temp[:,0]
        nxt_w = temp[:,1]

        t+=1
        jnt_err = calc_jnt_err(nxt_th, th_goal)
        dedt  = w * np.sign(jnt_err)
        th = nxt_th
        w = nxt_w

        if t >= t_limit:
            done = True
        elif np.all(abs(jnt_err) < thres): # no termination for vel
            done = True

        score += -np.linalg.norm(jnt_err)

    return th, w, score, t, True

def gen_rand_pos(quad): #,high):
    xy = (1*np.random.rand(3))
    if quad==2 or quad==3:
        xy[0] = -1*xy[0]
    if quad==3 or quad==4:
        xy[1] = -1*xy[1]
    mag = .4*rng.random() + .2
    goal = mag * .9999 * (xy/np.linalg.norm(xy)) + np.array([0,0,.3])
    if goal[2] < 0.05:
        goal[2] = 0.05
    
    return goal 


class action_space():
    def __init__(self):
        self.shape = np.array([3]) # three joint angles adjustments
        self.high = np.ones(3) * tau_max
        self.low = np.ones(3) * -tau_max

    # def sample(self):
    #     return 2*torch.rand(3) - 0.5

class observation_space():
    def __init__(self):
        self.shape = np.array([3])  # [x, y, dxdt, dydt, th1, th2, th3, w1, w2, w3]
                                    # [th1, th2, th3, w1, w2, w3]


class RobotEnv(object):
    def __init__(self, has_objects=True, num_obj=3, start=None,goal=None):
        self.robot = robot_3link()

        if has_objects:
            objs = []
            for i in range(num_obj):
                objs.append(rand_object(dt=dt))
            self.objs = objs
        else:
            self.objs = objs

        self.has_objects = has_objects
        self.num_obj = num_obj
        self.action_space = action_space()
        self.observation_space = observation_space()

        if isinstance(start, np.ndarray):
            self.start = start
            self.goal = goal
        else:
            quad1 = rng.choice(np.array([1,2,3,4]))
            quad2 = rng.choice(np.array([1,2,3,4]))
            while quad1 == quad2:
                quad2 = rng.choice(np.array([1,2,3,4]))
            s = gen_rand_pos(quad1)
            g = gen_rand_pos(quad2)
            th_arr = self.robot.reverse(goal=g)
            self.robot.set_pose(angle_calc(th_arr[:,0]))
            while self.robot.check_safety():
                g = gen_rand_pos(quad2)
                th_arr = self.robot.reverse(goal=g)
                self.robot.set_pose(angle_calc(th_arr[:,0]))
            self.goal = angle_calc(th_arr[:,0])
                
            th_arr = self.robot.reverse(goal=s)
            self.robot.set_pose(angle_calc(th_arr[:,0]))
            while self.robot.check_safety():
                s = gen_rand_pos(quad2)
                th_arr = self.robot.reverse(goal=s)
                self.robot.set_pose(angle_calc(th_arr[:,0]))
            self.start = angle_calc(th_arr[:,0])
            self.robot.set_pose(self.start)

        self.t_count = 0
        self.t_sum = 0
        self.done = False
        
    def env_replay(self, start_v:vertex, th_goal, obs_dict, steps):
        if not isinstance(th_goal, np.ndarray):
            th_goal = np.array(th_goal)
        th = np.array(start_v.th)
        w = start_v.w
        t_start = start_v.t
        return env_replay(th, w, t_start, th_goal, obs_dict, steps)