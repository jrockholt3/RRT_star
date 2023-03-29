from search_space import SearchSpace
from Robot_Env_v2 import RobotEnv, dt, t_limit, jnt_vel_max
from rrt_base import RRTBase, vertex
import numpy as np

class RRT_star(RRTBase):
    def __init__(self, X, start, goal, max_samples, r, n=5, steps=5):
        self.n = n
        self.steps = steps
        super().__init__(X, start, goal, max_samples, r)

    def cost_replan(self, v:vertex):
        """
        Go to root - generate a list of target and times along the way
        Edit the nodes along the path
         - remove the keys of the recalcuated nodes but have to still have them connected to thier children.. 
        """
        # go to root
        # generate targets and times
        # remove all nodes
        # travel to first target 
        # - add up cost
        # - 
        # replay path adding up costs along the way. 

    def rrt_search(self):
        v = vertex(self.start, 0)
        self.add_vertex(v)
        self.add_edge(v,v)
        converged = False
        loop_count = 0
        path = []
        while not converged:
            # sample a new node
            th_new = self.X.sample()
            # find n nearest nodes
            near = self.nearby(th_new, self.n)
            # try to reach new node from nearby nodes
            v_new = []
            for v in near:
                if v.t < 250:
                    th_fin, w_fin, reward, t_fin, flag = self.X.env.env_replay(v, th_new, self.X.obs_pos, self.steps)
                    if flag:
                        v_new.append((v,vertex(tuple(th_fin),t_fin,w=w_fin,reward=reward, targ=th_new)))
            # add the node with the best reward
            if len(v_new) == 0:
                continue
            (parent,v_best) = v_new.pop()
            best = np.linalg.norm(th_new - np.array(v_best.th))
            # best = v_best.reward
            for tup in v_new:
                p,v = tup[0],tup[1]
                err = np.linalg.norm(th_new - np.array(v.th))
                if err < best: # v.reward > best:
                    v_best = v
                    parent = p
                    best = np.linalg.norm(th_new - np.array(v_best.th))
                    # best = v.reward
            self.add_vertex(v_best)
            self.add_edge(v_best, parent)

            loop_count += 1
            if loop_count%100 == 0:
                print(loop_count,'checking for convergence', self.sample_count)
                converged = self.can_connect_to_goal()
                if converged: 
                    print('converged!')
                    path = self.reconstruct_path(self.start, self.goal)
                
            if loop_count > int(1e6) or self.sample_count > self.max_samples:
                converged = True
                v_a = self.get_nearest(self.goal)
                v_b = vertex(self.goal)
                if self.connect_with_pateince(30):
                    print('converged with patience')
                    path = self.reconstruct_path(self.start, self.goal)
                else:
                    print('failed to converge')

        return path



env = RobotEnv()
X = SearchSpace((750, np.pi, .9*np.pi, .9*np.pi), env)
start = tuple(env.start)
goal = tuple(env.goal)
print('goal', goal)
steps=20
r = jnt_vel_max*dt*10
max_samples = int(10000)

rrt = RRT_star(X, start, goal, max_samples, r,n=10,steps=steps)
path = rrt.rrt_search()
obs = rrt.get_obs()
rrt.plot_graph()
