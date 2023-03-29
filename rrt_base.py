import random 
import numpy as np
from search_space import SearchSpace
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Robot_Env import dt, j_max, tau_max, jnt_vel_max, calc_clip_vel
from support_classes import vertex, Tree

def steer(th1, th2, d):
    start, end = np.array(th1),np.array(th2)
    v = (end - start) / np.linalg.norm(end - start)
    steered_ptn = start + v*d
    return tuple(steered_ptn)

class RRTBase(object):
    def __init__(self, X:SearchSpace, start, goal, max_samples, r, prc=0.01):
        """
        X: search space
        start: begin robot pose (th1, th2, th3)
        goal: target robot pose (th1, th2, th3)
        max_samples: maximum of samples to take
        r: maximum allowable edge length (can be shorter)
        """
        self.X = X
        self.sample_count = 0
        self.max_samples = max_samples
        self.r = r
        self.prc = prc
        self.start = start
        self.goal = goal
        self.stop = False
        self.tree = Tree(3)

    def add_vertex(self, v:vertex):
        """
        insert vertex into tree
        """
        self.tree.V.insert(0, v.th, obj=v)
        self.tree.V_count += 1
        self.sample_count += 1

    def add_edge(self, child:vertex, parent:vertex):
        # connect parent to child
        self.tree.E[child.th] = parent

    def nearby(self, th, n):
        '''
        th: robot pose
        n: number of neighbors to return
        return list of neighbors as vertex
        '''
        return self.tree.V.nearest(th, num_results=n,objects="raw")

    def get_nearest(self, th):
        """
        return nearest vertex
        """
        return next(self.nearby(th,1))

    def new_and_near(self):
        # q = length of edge when steering
        # returns the joint angles of near vertex

        th_rand = self.X.sample()
        v_nearest = self.get_nearest(th_rand)
        r = np.linalg.norm(np.array(v_nearest.th) - np.array(th_rand))
        if r < self.r:
            th_new = steer(v_nearest.th, th_rand, self.r)
        else:
            th_new = th_rand
        
        x = self.X.gen_car_ptns(th_new, v_nearest.t+1) # points along pose at time t
        if not self.tree.V.count(th_new) == 0 or not self.X.obstacle_free(x):
            return None, None
        self.sample_count += 1
        return th_new, v_nearest

    def connect_to_point(self, v_a:vertex, v_b:vertex):
        """
        connect vertex a to vertex b
        solve for tau, w, and th of vertex b
        """
        v_b.t = v_a.t + 1
        if self.tree.V.count(v_b.th) == 0:
            r = np.linalg.norm(np.array(v_b.th) - np.array(v_a.th))
            r_max = v_a.r_max(v_b.th)
            if r < r_max:
                self.add_vertex(v_b)
                self.add_edge(v_b, v_a)
            else:
                v_b.th = steer(v_a.th, v_b.th, r_max)
                self.add_vertex(v_b)
                self.add_edge(v_b, v_a)
            return True
        return False


    def can_connect_to_goal(self):
        # check if we can connect
        v_nearest = self.get_nearest(self.goal)
        if self.goal in self.tree.E and v_nearest.th in self.tree.E[self.goal]:
            # goal is already connected 
            return True
        if np.linalg.norm(np.array(v_nearest.th) - self.goal) <= self.r:
            v_goal = vertex(self.goal, v_nearest.t+1)
            self.add_vertex(v_goal)
            self.add_edge(v_goal, v_nearest)
            return True
        return False

    def connect_with_pateince(self, tries):
        v_nearest = self.get_nearest(self.goal)
        if np.linalg.norm(np.array(v_nearest.th) - self.goal) <= self.r:
            t = v_nearest.t
            for i in range(tries):
                if self.X.collision_free(v_nearest.th, self.goal, t+i, t+i+1):
                    v_goal = vertex(self.goal, v_nearest.t+i+1)
                    self.add_vertex(v_goal)
                    self.add_edge(v_goal, v_nearest)
                    return True
        return False

    def reconstruct_path(self, start, goal):
        '''
        start: (th1, th2, th3)
        goal: (th1, th2, th3)
        '''
        # path = [goal]
        path = []
        curr = goal
        if start == goal:
            return path
        parent = self.tree.E[curr]
        while not parent.th == start:
            path.append((parent.t,parent.th,parent.targ))
            curr = parent.th
            parent = self.tree.E[curr]
        path.append((0,start,np.zeros(3)))

        path.reverse() 
        
        return path

    def get_obs(self):
        return self.X.obs_pos

    def plot_graph(self, add_path=False, path=None):
        th_arr = []
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for k in self.tree.E.keys():
            arr = np.array(k)
            th_arr.append(arr)
            parent = self.tree.E[k]
            xx = np.array([parent.th[0], k[0]])
            yy = np.array([parent.th[1], k[1]])
            zz = np.array([parent.th[2], k[2]])
            ax.plot3D(xx,yy,zz,'k',alpha=.1)

        th_arr = np.vstack(th_arr)
        xx = th_arr[:,0]
        yy = th_arr[:,1]
        zz = th_arr[:,2]

        ax.scatter3D(xx,yy,zz,alpha=.2)
        ax.scatter3D(self.start[0], self.start[1], self.start[2], 'r')
        ax.scatter3D(self.goal[0], self.goal[1], self.goal[2], 'g')

        if add_path:
            xx,yy,zz = [],[],[]
            for th_tup in path:
                xx.append(th_tup[0])
                yy.append(th_tup[1])
                zz.append(th_tup[2])

            ax.plot3D(xx,yy,zz, 'r')

        plt.show()

