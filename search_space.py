import numpy as np
from rtree import index
from Robot_Env import RobotEnv
import Robot_Env
import uuid

def gen_obstacles(env:RobotEnv, obs_index:index.Index):
    '''
    have to generate the min and max corners of obstacles at 
    each time step
    '''
    objs = env.objs.copy()
    dt = Robot_Env.dt
    t_limit = Robot_Env.t_limit
    min_prox = Robot_Env.min_prox
    obs = []

    t = 0
    time_steps = int(np.ceil(t_limit/dt))
    while t <= time_steps:
        for o in objs:
            center = o.curr_pos
            x,y,z = center[0],center[1],center[2]
            obs_i = (t, x-min_prox, y-min_prox, z-min_prox, t, x+min_prox, y+min_prox, z+min_prox)
            obs_index.add(uuid.uuid4(), obs_i)
            obs.append(obs_i)
            o.step()
        t += 1

    return obs


class SearchSpace(object):
    def __init__(self, dimension_lengths, env:RobotEnv):
        self.env = env
        self.dimension_lengths = dimension_lengths
        self.dimension = len(dimension_lengths)
        p = index.Property()
        p.dimension = self.dimension

        obs = index.Index(interleaved=True, properties=p)
        gen_obstacles(env,obs)
        self.obs = obs

        self.jnt_max = env.robot.jnt_max.copy()
        self.jnt_min = env.robot.jnt_min.copy()
        self.n = 10

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: locations along the robotic arm to check (t,x,y,z)
        :return: True if not inside an obstacle, False otherwise
        """
        for x_ in x:
            if self.obs.count(x_) != 0:
                return False

        return True 

    def sample_free(self, t):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            th = self.sample()
            x = self.gen_car_ptns(th, t) # x = (t,x,y,z)
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end, t1, steps):
        """
        Check if a line segment intersects an obstacle
        :param start: start pose of the robot as a tup
        :param end: ending pose of the robot as a tup
        :param t1, t2: time steps for start and end
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        t2 = t1 + steps
        x = []
        curr = np.array(start)
        u = (curr - np.array(end))/steps
        for t in range(t1,t2):
            x.append(self.gen_car_ptns(curr,t))
            curr = curr + u * (t-t1)
        x.append(self.gen_car_ptns(end,t2))

        for x_ in x:
            if not self.obstacle_free(x_):
                return False
        return True

    def sample(self):
        """
        Return a random joint position
        """

        th = np.random.uniform(self.jnt_min, self.jnt_max)
        self.env.robot.set_pose(th)
        while self.env.robot.check_safety():
            th = np.random.uniform(self.jnt_min, self.jnt_max)
            self.env.robot.set_pose(th)

        return tuple(th)

    def gen_car_ptns(self, th, t):
        '''
        generate points along robot arm to test for collsion
        th: joint angles of the robot
        t: time step the robot occupies 
        returns a list with n # of points along each robot link
        '''
        if not isinstance(th, np.ndarray):
            th = np.array(th)
        n = self.n
        x_list = []
        ptns = self.env.robot.forward(th)
        for i in range(0,ptns.shape[1]-1):
            x1 = ptns[0:3,i]
            x2 = ptns[0:3,i+1]
            u = (x2 - x1)
            for i in range(1, n+1):
                arr = u*i/n + x1
                new = (t, arr[0], arr[1], arr[2])
                x_list.append(new)

        return x_list



