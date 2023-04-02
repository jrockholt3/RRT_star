import random 
import numpy as np
from search_space import SearchSpace
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from Robot_Env import dt, j_max, tau_max, jnt_vel_max, calc_clip_vel
from env_config import dt, j_max, tau_max, jnt_vel_max
from Robot_Env_v2 import env_replay
from support_classes import vertex, Tree

def steer(th1, th2, d):
    start, end = np.array(th1),np.array(th2)
    v = (end - start) / np.linalg.norm(end - start)
    steered_ptn = start + v*d
    return tuple(steered_ptn)

class RRTBase(object):
    def __init__(self, X:SearchSpace, start, goal, max_samples, r, d, prc=0.01):
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
        self.d = d
        self.prc = prc
        self.start = start
        self.goal = goal
        self.stop = False
        self.tree = Tree(3)
        self.edge_count = 0

    def add_vertex(self, v:vertex):
        """
        insert vertex into tree
        """
        self.tree.V.insert(v.id, v.th, obj=v)
        self.tree.V_count += 1
        self.sample_count += 1
    
    def delete_vertex(self, v:vertex):
        """
        deletes the vertex in the index
        """
        self.tree.V.delete(v.id, v.th)
        self.tree.V_count = self.tree.V_count - 1
        self.sample_count = self.sample_count - 1

    def add_edge(self, child:vertex, parent:vertex):
        # connect parent to child
        self.edge_count += 1
        child.id = self.edge_count
        self.tree.E[child.id] = parent
        return child 
        
    def init_root(self, start:vertex):
        self.tree.E[start.id] = start

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

    def get_win_radius(self, center, r):
        top = center + r
        bottom = center - r
        return list(self.tree.V.intersection())


    def steer(self, th1, th2):
        d = self.d
        start, end = np.array(th1),np.array(th2)
        norm = np.linalg.norm(end - start)
        if norm >= d:
            v = (end - start) / norm
            steered_ptn = start + v*d
            return tuple(steered_ptn)
        else:
            return th2
        return th2

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
            return v_nearest, True
        return None, False

    def connect_with_pateince(self, tries):
        '''
        Currently not in uses
        '''
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

    def reconstruct_path(self, curr, start, goal):
        '''
        start: (th1, th2, th3)
        goal: (th1, th2, th3)
        '''
        # path = [goal]
        path = []
        traj = []
        path.append(goal)
        if start == goal:
            return path
        while not curr.id == 0:
            # path.append((curr.t,curr.th,curr.targ))
            print('adding', curr.th)
            path.append(curr.th)
            traj.append((curr.t, curr.targ))
            curr = self.tree.E[curr.id]
        path.append(start)

        path.reverse() 
        
        return path, traj
    
    def recalc_path(self, leaf:vertex):
        '''
        starts at a leaf then finds all nodes along path to root
        starting at root, it replays the sequence of targets given by the path
        once a new th is calculated for a node, that node is returned 
        '''
        curr_v = leaf.copy()
        path = []
        while not curr_v.th == self.start:
            path.append(curr_v)
            curr_v = self.tree.E[curr_v.id]
        curr_v = self.tree.E[0] # add the root to the path
        path.append(curr_v)
        path.reverse

        t = 0 
        new_path = []
        curr_v = path.pop()
        new_path.append(curr_v)
        while len(path) > 0:
            nxt_v = path.pop()
            th = np.array(curr_v.th)
            steps = nxt_v.t - curr_v.t
            nxt_th, w, score, t, flag = env_replay(th, curr_v.w, t, curr_v.targ, self.X.obs_pos, steps)
            nxt_v.th = tuple(nxt_th)
            nxt_v.w = w
            nxt_v.reward = score
            nxt_v.t = t
            self.tree.E[nxt_v.id] = curr_v # edit the edges
            self.delete_vertex(nxt_v) # deletes the vertex
            self.add_vertex(nxt_v) # re-adds the vertex with the new information
            curr_v = nxt_v 

        if curr_v.id == leaf.id:
            return curr_v
        else:
            return False

    def reward_calc(self, leaf:vertex):
        curr_v = leaf
        reward = 0
        while not curr_v.th == self.start:
            reward += curr_v.reward
            curr_v = self.tree.E[curr_v.id]
        return reward

    def get_obs(self):
        return self.X.obs_pos

    def plot_graph(self, every=10, add_path=False, path=None):
        # entres = np.empty(len(self.tree.E),dtype=tuple)
        # for k in self.tree.E.keys():
        #     parent = self.tree.E[k]
        #     entres[parent.id] = parent.th

        th_arr = []
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        i = 0
        for k in self.tree.E.keys():
            if i%every == 0:
                parent = self.tree.E[k]
                arr = np.array(parent.th)
                th_arr.append(arr)
                # child_th = entres[k]
                # # parent = self.tree.E[k]
                # xx = np.array([parent.th[0], child_th[0]])
                # yy = np.array([parent.th[1], child_th[0]])
                # zz = np.array([parent.th[2], child_th[0]])
                # ax.plot3D(xx,yy,zz,'k',alpha=.1)

        th_arr = np.vstack(th_arr)
        xx = th_arr[:,0]
        yy = th_arr[:,1]
        zz = th_arr[:,2]

        ax.scatter3D(xx,yy,zz,alpha=.1)
        ax.scatter3D(self.start[0], self.start[1], self.start[2], 'r')
        ax.scatter3D(self.goal[0], self.goal[1], self.goal[2], 'g')

        if add_path:
            xx,yy,zz = [],[],[]
            for th in path:
                xx.append(th[0])
                yy.append(th[1])
                zz.append(th[2])

            ax.plot3D(xx,yy,zz, 'r')

        plt.show()

