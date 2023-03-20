from search_space import SearchSpace
from Robot_Env import RobotEnv, dt, t_limit, jnt_vel_max
from rrt_base import RRTBase, vertex
import numpy as np

class RRT_star(RRTBase):
    def __init__(self, X, start, goal, max_samples, r):

        super().__init__(X, start, goal, max_samples, r)

    def rrt_search(self):
        v = vertex(self.start, 0)
        self.add_vertex(v)
        self.add_edge(v,v)
        converged = False
        loop_count = 0
        path = []
        while not converged:
            th_new, v_nearest = self.new_and_near()

            if th_new is None:
                continue
            
            new_v = vertex(th_new)
            flag = self.connect_to_point(v_nearest, new_v)
            # flag = self.connect_with_pateince(v_nearest, new_v, 10)

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




# testing
env = RobotEnv()
print(tuple(env.robot.jnt_max))
X = SearchSpace((750, np.pi, .9*np.pi, .9*np.pi), env)
start = tuple(env.start)
goal = tuple(env.goal)
r = jnt_vel_max*dt*10
max_samples = int(5e3)

rrt = RRT_star(X, start, goal, max_samples, r)
path = rrt.rrt_search()
print(path)
rrt.plot_graph(add_path=True, path=path)
