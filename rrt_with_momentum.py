import Robot_Env
from Robot_Env import RobotEnv, dt, t_limit, tau_max, jnt_vel_max
import numpy as np 
from search_space import SearchSpace
from rrt_base import vertex, Tree, RRTBase

class RRT_WM(RRTBase):
    def __init__(self, X:SearchSpace, start, goal, max_samples, r, prc=0.01)
        super().__init__(X, start, goal, max_samples, r)

    def search(self):
        v = vertex(self.start,0)
        self.add_vertex(v)
        self.add_edge(v,v)
        converged = False
        loop_count = 0
        path = []

        while not converged:
            th_new, n_near = self.
