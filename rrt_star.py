from search_space import SearchSpace
from Robot_Env_v2 import RobotEnv, dt, t_limit, jnt_vel_max
from rrt_base import RRTBase
from support_classes import vertex
import numpy as np

class RRT_star(RRTBase):
    def __init__(self, X, start, goal, max_samples, r, d,  thres, n=5, steps=5):
        self.n = n
        self.steps = steps
        self.thres = thres
        super().__init__(X, start, goal, max_samples, r, d)

    def get_reachable(self, v, targ):
        # print(v.t)
        if v.t < t_limit/dt:
            targ_i = self.steer(v.th, targ)
            th_fin, w_fin, reward, t_fin, flag = self.X.env.env_replay(v, targ_i, self.X.obs_pos, self.steps)
            return vertex(tuple(th_fin), t_fin, w_fin, reward, targ_i), flag
        else:
            return None, False

    def add_nearest(self, pv_pairs, targ):
        '''
        add node nearest to the target
        '''
        # (parent,v_best) = pv_pairs.pop()
        best = np.inf
        # best = v_best.reward
        flag = False
        for tup in pv_pairs:
            p,v = tup[0],tup[1]
            err = np.linalg.norm(targ- np.array(v.th))
            d = np.linalg.norm(np.array(p.th) - np.array(v.th))
            if d >= self.thres:
                if err < best: # v.reward > best:
                    flag = True
                    v_best = v
                    parent = p
                    best = np.linalg.norm(targ- np.array(v_best.th))
                    # best = v.reward
        if flag:
            child = self.add_edge(v_best, parent)
            self.add_vertex(child)
            return child, flag
        else:
            return None, flag
    
    def add_highest_reward(self, pv_pairs):
        '''
        add the parent-leaf pair with the highest
        reward
        '''
        flag = False
        best_reward = -np.inf
        for tup in pv_pairs:
            p,v = tup[0],tup[1]
            d = np.linalg.norm(np.array(p.th) - np.array(v.th))
            reward = self.reward_calc(p) + v.reward
            if d >= self.thres and reward > best_reward:
                flag = True
                best_reward = reward
                parent = p
                v_best = v
        
        if flag:
            child = self.add_edge(v_best, parent)
            self.add_vertex(child)
            return child, flag
        else:
            return None, flag
    
    def add_all(self, pv_pairs):
        '''
        add all edges in list
        '''
        for tup in pv_pairs:
            p,v = tup[0],tup[1]
            d = np.linalg.norm(np.array(p.th) - np.array(v.th))
            if d >= thres: 
                child = self.add_edge(v,p)
                self.add_vertex(child)

    
    # def get_v_in_radius(self, v):
    #     min_th = v.th - self.r
    #     max_th = v.th + self.r
    #     v_in_r = self.tree.V.intersection()

    def rewire(self, v, v_near):
        parent = self.tree.E[v.id]
        reward = self.reward_calc(v)
        for v_i in v_near:
            if parent.id == v_i.id or v.id == v_i.id:
                continue
            else:
                v_i = self.recalc_path(v_i)
                v_reached, flag = self.get_reachable(v, v_i.th)
                reward_i = self.reward_calc(v_i)
                if reward_i < reward + v_reached.reward and flag:
                    v_reached.id = v_i.id
                    self.delete_vertex(v_i)
                    self.add_vertex(v_reached)
                    self.tree.E[v_reached.id] = v

    def rrt_search(self):
        v = vertex(self.start)
        self.init_root(v)
        self.add_vertex(v)
        converged = False
        loop_count = 0
        path = []
        traj=[]
        v = self.tree.E[0]
        for i in range(100):
            th_new = self.X.sample()
            v_reached, flag = self.get_reachable(v,th_new)
            if flag:
                child = self.add_edge(v_reached,v)
                self.add_vertex(child)
        
        while not converged:
            # sample a new node
            if loop_count%5 == 0:
                th_new = self.goal # bias
            else:
                th_new = self.X.sample() # explore 
            # find n nearest nodes
            near = self.nearby(th_new, self.n)
            # try to reach new node from nearby nodes
            v_new = []
            for v in near:
                # v = self.recalc_path(v) # update v's position
                v_reached,flag = self.get_reachable(v, th_new)
                if flag:
                    v_new.append((v,v_reached))
            # add the node with the best reward
            if len(v_new) == 0:
                continue
            
            # tests n different nearest nodes to the original target, th_new
            # adds the node that is the clostest to the target reachable
            # from the n_nodes list in v_new
            # v, v_added_flag = self.add_nearest(v_new, th_new)
            v, v_added_flag = self.add_highest_reward(v_new)
            # self.add_all(v_new)

            # parents = []
            # curr = v.copy()
            # look_back = 5
            # for i in range(look_back):
            #     parents.append(self.tree.E[curr.id])

            # now we need to rewire the newly added node
            # if v_added_flag:
            #     if v.id in self.tree.E:
            #         # near = self.get_win_radius(v.th, jnt_vel_max*self.steps*dt)
            #         # print('within radius',jnt_vel_max*self.steps*dt,'there are', len(near))
            #         near = self.nearby(v.th, 5)
            #         self.rewire(v, near)

            loop_count += 1
            if loop_count%100 == 0:
                print(loop_count,'checking for convergence', self.sample_count)
                v, converged = self.can_connect_to_goal()
                if converged: 
                    print('converged!')
                    path,traj = self.reconstruct_path(v, self.start, self.goal)
                
            if loop_count>=max_samples:
                v_a = self.get_nearest(self.goal)
                v_b = vertex(self.goal)
                converged = True
                if self.connect_with_pateince(30):
                    print('converged with patience')
                    path,traj = self.reconstruct_path(self.start, self.goal)
                else:
                    print('failed to converge')

        return path, traj



env = RobotEnv(num_obj=3)
X = SearchSpace((750, np.pi, .9*np.pi, .9*np.pi), env)
start = tuple(env.start)
goal = tuple(env.goal)
print('goal', goal)
steps = 10
thres = np.linalg.norm([.003,.003,.003])*(steps/25)
n = 25
r = np.linalg.norm(jnt_vel_max*dt*steps*np.ones(3)/5)
d = np.linalg.norm(jnt_vel_max*dt*steps*1.5*np.ones(3))
max_samples = int(3000)
rrt = RRT_star(X, start, goal, max_samples, r, d, thres,n=n,steps=steps)
path,traj = rrt.rrt_search()
print(path)
rrt.plot_graph(every=1.0,add_path=True, path=path)
