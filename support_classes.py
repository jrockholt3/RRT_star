import random 
import numpy as np
from rtree import index

class vertex(object):
    def __init__(self, th, t = 0, w=np.zeros(3),reward=0,dt=.016, targ=np.zeros(3)):
        self.th = th # (th1,th2,th3) tuple
        self.t = t
        self.reward = reward
        self.w = w
        self.tau = np.zeros_like(self.th)
        self.dt = dt
        self.targ = targ

    # def r_max(self, th2, steps):
    #     '''
    #     th2 = next goal pose (th1,th2,th3) as a tuple 
    #     returns: maximum distance along (th2-th1) vector based on max jerk
    #     '''
    #     th2 = np.array(th2)
    #     t = steps * self.dt
    #     # maximum reachable change based on max jerk
    #     J = (th2 - (self.tau * (t**2/2) + self.w * (t) + self.th)) / (t**3/6)
    #     J = np.clip(J, -j_max, j_max)
    #     temp = J * (t**3/6) + self.tau * (t**2/2) + self.w * t + self.th
    #     r_max_j = np.linalg.norm(temp - self.th)
    #     # max reachable change based on max torque
    #     tau = (th2 - self.th1 - self.w*t) / (t**2/2)
    #     tau = np.clip(tau, -tau_max, tau_max)
    #     temp = tau*(t**2/2) + self.w*t + self.th
    #     r_max_tau = np.linalg.norm(temp - self.th)
    #     # max reachable change based on max velocity
    #     w = (th2 - self.th)/t
    #     w = np.clip(w, -jnt_vel_max, jnt_vel_max)
    #     temp = w*t + self.th
    #     r_max_w = np.linalg.norm(temp - self.th)
    #     return np.min([r_max_j, r_max_tau, r_max_w])
    
    # def stoping_point(self):
    #     '''
    #     Return the position the arm would be after an emergency stop
    #     '''



class Tree(object):
    def __init__(self, dims):
        '''
        Tree 
        dims: dimension of storage space
        X: search space
        V: joint-space spatial storage of vertex's
        E: dictionary of edges 
        '''
        p = index.Property()
        p.dimension = dims
        self.V = index.Index(interleaved=True, properties=p)
        self.V_count = 0
        self.E = {}