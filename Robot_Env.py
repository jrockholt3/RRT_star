import numpy as np
import Robot3D as Robot
from math import atan2
from Object_v2 import rand_object
# import torch 
from Robot3D import workspace_limits
from support_classes import vertex
# global variables
dt = 0.016 # time step
t_limit = 4 # time limit in seconds
thres = np.array([.03, .03, .03]) # joint error threshold
vel_thres = thres # joint velocity error threshold for stopping
# weighting different parts of the reward function
scale = 1/(t_limit/dt)
Alpha = .8
Beta = 0 # positive reward for proximity
Gamma = 0 # negative rewards for be jumps in torque (minimize jerk)
prox_thres = .05 # proximity threshold - 5 cm
min_prox = 0.1
vel_prox = 0.5

# Controller gains
tau_max = 30 #J*rad/s^2, J = 1
damping = tau_max*.5
P = 500
D = 0
P_v = 10.0
D_v = 5.0
jnt_vel_max = np.pi # rad/s
j_max = tau_max/dt # maximum allowable jerk on the joints

rng = np.random.default_rng()

def a(list):
    return np.array(list)

# robot = Robot.robot_3link()
# pos1 = robot.reverse(a([-.6,.1]),np.pi/4)
# pos1 = pos1[:,1]
# robot.forward(th=pos1,make_plot=True)
# pos2 = robot.reverse(a([.6,.1]), 2*np.pi/3)
# pos2 = pos2[:,0]
# robot.forward(th=pos2,make_plot=True)
# pos3 = robot.reverse(a([.5,-.3]), 3*np.pi/4)
# pos3 = pos3[:,1]
# robot.forward(th=pos3,make_plot=True)
# pos4 = robot.reverse(a([-0.6,-0.6]), np.pi/3)
# pos4 = pos4[:,0]
# robot.forward(th=pos4,make_plot=True)
# XX = a([pos1.copy(), pos2.copy(), pos3.copy(), pos4.copy()])
# del robot

# def calc_jnt_err(curr,goal):
#     # angle error is defined in the right-hand positive sense from th1 to th2
#     # th1 is the current position of the robot, th2 is the goal
    
#     th1 = curr
#     th2 = goal
#     x1 = np.cos(th1)
#     x2 = np.cos(th2)
#     y1 = np.sin(th1)
#     y2 = np.sin(th2)

#     r1 = np.array([x1,y1])
#     r2 = np.array([x2,y2])
#     s = np.zeros(3,dtype=np.float64)
#     c = np.zeros(3,dtype=np.float64)
#     for i in range(r1.shape[1]):
#         arr = np.cross(r1[:,i], r2[:,i])
#         s[i] = arr
#         c[i] = np.dot(r1[:,i], r2[:,i]) 
        
#     arr = np.zeros(3,dtype=np.float64)
#     for i in range(len(s)):
#         th = atan2(s[i],c[i])
#         if np.round(th,5) == -np.round(np.pi,5):
#             th = np.pi
#         arr[i] = th
    
#     return arr

def calc_jnt_err(curr,goal):
    # first two 
    s1 = np.sin(curr)
    c1 = np.cos(curr)
    curr = np.arctan2(s1,c1)
    s2 = np.sin(goal)
    c2 = np.cos(goal)
    goal = np.arctan2(s2,c2)

    # th1 = curr[2]
    # th2 = goal[2]
    # x1 = np.cos(th1)
    # y1 = np.sin(th1)
    # x2 = np.cos(th2)
    # y2 = np.sin(th2)

    # r1 = np.array([x1,y1])
    # r2 = np.array([x2,y2])
    # s3 = np.cross(r1,r2)
    # c3 = np.dot(r1,r2)
    # th = np.arctan2(s3,c3)

    jnt_err = goal - curr
    # jnt_err[2] = th

    return jnt_err


def angle_calc(th):
    s = np.sin(th)
    c = np.cos(th)
    arr = np.arctan2(s,c)
    return arr

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

def calc_clip_vel(prox):
    return jnt_vel_max * (1 - np.exp(-(2/3)*(prox-min_prox)/(vel_prox-min_prox)**2))

class PDControl():
    def __init__(self, P=P, D=D, tau_max=tau_max, dt=dt, damping=damping):
        self.P = P
        self.D = D
        self.jnt_err = np.array([0,0,0])
        self.prev_jnt_err = np.array([0,0,0])
        self.tau_mx = tau_max
        self.J = 1
        self.dt = dt

    def step(self, jnt_err):
        dedt = (jnt_err - self.jnt_err)/self.dt
        self.jnt_err = jnt_err
        tau = self.P*self.jnt_err + self.D*dedt
        tau = np.clip(tau, -self.tau_mx, self.tau_mx)
        return tau, dedt

class VelControl():
    def __init__(self, P=P_v, D=D_v, tau_max=tau_max,dt=dt):
        self.P = P
        self.D = D
        self.vel_err = np.array([0,0,0])
        self.tau_max = tau_max
        self.dt = dt

    def step(self, vel_err):
        dedt = (vel_err - self.vel_err)
        self.vel_err = vel_err
        tau = self.P*self.vel_err + self.D*dedt
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau, dedt

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
    # have to generate random poses
    def __init__(self, eval=False, has_objects=True, close_by=False, ds = .1, num_obj=3,
                    start=None, goal=None):
        self.robot = Robot.robot_3link()

        if has_objects:
            objs = []
            for i in range(num_obj):
                objs.append(rand_object(dt=dt))
            self.objs = objs
        else:
            self.objs = [] 
        self.num_obj = num_obj
        self.action_space = action_space()
        self.observation_space = observation_space()
        self.reward_range = [0, -np.inf]
        self.Controller = PDControl(dt=dt)
        self.vel_control = VelControl(dt=dt)
        self.eval = eval
        self.has_objects = has_objects
        self.close_by = close_by
        self.ds = ds

        # setting runtime variables 
        # setting start and end positions
        if isinstance(start, np.ndarray):
            # define start and end goals
            self.start = start
            self.goal = goal
        else:
            # define random goal
            quad1 = rng.choice(np.array([1,2,3,4]))
            s = gen_rand_pos(quad1)
            if self.close_by:
                g = gen_rand_pos(quad1)
            else:
                quad2 = rng.choice(np.array([1,2,3,4]))
                while quad1 == quad2:
                    quad2 = rng.choice(np.array([1,2,3,4]))
                g = gen_rand_pos(quad2)

            # get a joint position that doesn't have a collison
            rand_int = 0 #rng.choice([0,1])
            th_arr = self.robot.reverse(goal=g)
            self.goal = angle_calc(th_arr[:,rand_int])
            self.robot.set_pose(self.goal)
            if self.robot.check_safety():
                g = gen_rand_pos(quad2)
                th_arr = self.robot.reverse(goal=g)
                self.goal = angle_calc(th_arr[:,rand_int])
            # get a joint position that doesn't have a collison
            # rand_int = rng.choice([0,1])
            th_arr = self.robot.reverse(goal=s)
            self.start = th_arr[:,rand_int]
            self.robot.set_pose(self.start)
            if self.robot.check_safety():
                # rand_int = (rand_int+1)%2
                s = gen_rand_pos(quad1)
                th_arr = self.robot.reverse(goal=s)
                self.start = angle_calc(th_arr)
                self.robot.set_pose(self.start)

        # goal = np.array([-.3,-.3,.2])*
        # th = self.robot.reverse(goal=goal)
        # self.start = th[:,1]
        # self.robot.set_pose(self.start)
        # self.goal = np.array([-np.pi/4, -np.pi/4, -np.pi/4])

        self.done = False
        self.jerk_sum = 0
        self.t_sum = 0
        self.t_count = 0
        self.info = {}
        self.jnt_err = calc_jnt_err(self.robot.pos, self.goal) 
        self.jnt_err_vel = np.array([0,0,0])
        self.prev_tau = np.array([0,0,0])

    def env_replay(self, start_v:vertex, th_goal, obs_dict, steps):
        '''
        input
        start_v: a vertex describing starting position and velocity
        th_gaol: np.array [th1,th2,th3], a sampled position in the joint space
        obs_arr: dictionary of object position, t=key, entries = np.array of n positions 3xn
        steps: amount of times steps to look ahead - if a goal is reached in less time steps stops
        '''
        # setting state variables
        self.robot.set_pose(start_v.th)
        self.robot.set_jnt_vel(start_v.w)
        self.start = np.array(start_v.th)
        self.goal = th_goal
        self.jnt_err = calc_jnt_err(self.start, self.goal)
        self.t_count = start_v.t

        # score = 0
        done = False
        t=start_v.t
        if t >= t_limit/dt:
            reward = -np.inf
            return self.robot.pos, self.robot.jnt_vel, reward, t, False
        while not done and t<start_v.t + steps and t < t_limit/dt:
            _, reward, done, info = self.step(np.zeros(3), use_PID=True, eval=True, obs_arr=obs_dict[t],use_v_thres=False)
            # score += reward
            t+=1

        
        return self.robot.pos, self.robot.jnt_vel, reward, t, True


    def step(self, action, use_PID=False, eval=False, use_VControl=False, w=None, obs_arr=None, use_v_thres=True):
        self.t_count += 1
        # stopping the robot is object is too close
        paused = False
        prox = np.inf
        if isinstance(obs_arr, np.ndarray):
            # passing in pre-defined obs pos
            for i in range(0,obs_arr.shape[1]):
                temp = np.min(self.robot.proximity(obs_arr[:,i]))
                if temp<prox:
                    prox = temp
        else:
            for o in self.objs:
                temp = np.min(self.robot.proximity(o.curr_pos)[1:3])
                if temp < prox:
                    prox = temp
                o.step()
        
        # stoping the robot is minimum proximity is reached (aka collison)
        if prox <= min_prox:
            paused = True 

        # scaling velocity based on proximity 
        if prox <= (vel_prox):
            clip_val = calc_clip_vel(prox)
        else:
            clip_val = jnt_vel_max

        # iterating the robot pos + speed
        tau = self.prev_tau
        safety_violation = False
        if not paused:
            if use_PID:
                tau, dedt = self.Controller.step(self.jnt_err)
            elif use_VControl:
                tau, dedt = self.vel_control.step(w - self.robot.jnt_vel)
            else:
                if not isinstance(action,np.ndarray):
                    action = action.cpu()
                    action = action.detach()
                    action = action.numpy()
                action = action.reshape(3) 
                tau = action
            self.prev_tau = tau
            nxt_vel = (tau-damping*self.robot.jnt_vel)*dt + self.robot.jnt_vel
            nxt_vel = np.clip(nxt_vel, -clip_val, clip_val)
            nxt_pos = angle_calc((tau-damping*self.robot.jnt_vel) * (dt**2/2) + self.robot.jnt_vel*dt + self.robot.pos)
            
            self.robot.set_jnt_vel(nxt_vel)
            self.robot.set_pose(nxt_pos)
            safety_violation = self.robot.check_safety()

            jnt_err = calc_jnt_err(self.robot.pos, self.goal) 
            self.jnt_err_vel = (self.jnt_err - jnt_err)/dt
            self.jnt_err = jnt_err 
        else:
            self.robot.set_jnt_vel(np.array([0,0,0]))
            self.jnt_err_vel = np.array([0,0,0])

        # check for termination, update bonus and reward
        bonus = 0
        done = False
        self.t_sum = self.t_count*dt
        if self.t_sum >= t_limit:
            done = True
            # bonus = 2*np.exp(-np.dot(self.jnt_err,self.jnt_err))
        elif np.all(abs(self.jnt_err) < thres):
            if use_v_thres:
                if np.all(abs(self.jnt_err_vel) < vel_thres): 
                    done = True 
                    bonus = 2*np.exp(-np.dot(self.jnt_err,self.jnt_err)-np.dot(self.jnt_err_vel, self.jnt_err_vel))
                    # print('finished by converging!!')
            else:
                done = True 
                bonus = 2*np.exp(-np.dot(self.jnt_err,self.jnt_err)-np.dot(self.jnt_err_vel, self.jnt_err_vel))
                # print('finished w/out vel')
        else:
            done = False

        safety_bonus = 0
        # if safety_violation:
        #     safety_bonus = 10
        #     done = True

        # reward = -Alpha * np.sum(np.abs(self.jnt_err)) + bonus 
        reward = -np.linalg.norm(self.jnt_err) 

        # collecting point cloud data and creating state
        coords = []
        feats = []
        if not eval: # skip over computation when making an animation
            rob_coords, rob_feats = self.robot.get_coords(self.t_count)
            for obj in self.objs:
                c,f = obj.get_coords(self.t_count)
                coords.append(c)
                feats.append(f)
            coords.append(rob_coords)
            feats.append(rob_feats)
            coords = np.vstack(coords)
            feats = np.vstack(feats)
        state = (coords,feats,self.robot.pos,self.goal)
        if use_PID or use_VControl:
            self.info = {'tau': tau}
        return state, reward, done, self.info


    def reset(self):
        new_env = RobotEnv(eval=self.eval,has_objects=self.has_objects,close_by=self.close_by,ds=self.ds,num_obj=self.num_obj)
        
        coords = []
        feats = []
        for obj in new_env.objs:
            c,f = obj.get_coords(new_env.t_count)
            coords.append(c)
            feats.append(f)
        rob_coords, rob_feats = new_env.robot.get_coords(t=new_env.t_count)
        coords.append(rob_coords)
        feats.append(rob_feats)
        coords = np.vstack(coords)
        feats = np.vstack(feats)

        state = (coords,feats,self.robot.pos,self.goal)
        return new_env, state

