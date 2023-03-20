import Robot_Env
from Robot_Env import dt, t_limit
from utils import gen_obstacles
import numpy as np
from rtree import index
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

env = Robot_Env.RobotEnv()

print(np.ceil(t_limit/dt)*3)

p = index.Property()
p.dimension = 4
obs = index.Index(interleaved=True, properties=p)
obs_list = gen_obstacles(env,obs)

th = np.random.uniform(env.robot.jnt_min, env.robot.jnt_max)
env.robot.set_pose(th)
while env.robot.check_safety():
    th = np.random.uniform(env.robot.jnt_min, env.robot.jnt_max)
    env.robot.set_pose(th)

ptns = env.robot.forward()
print(ptns)
res = .01
n = 10
t = 30
arr = []
for i in range(1,ptns.shape[1]-1):
    x1 = ptns[0:3,i]
    x2 = ptns[0:3,i+1]
    u = (x2 - x1)
    for i in range(1,n+1):
        new = u*i/n + x1
        tup = (t, new[0], new[1], new[2])
        arr.append(tup)

for x_ in arr:
    print(obs.count(x_)==0)
# arr = np.vstack(arr)
# xx = arr[:,0]
# yy = arr[:,1]
# zz = arr[:,2]
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(xx,yy,zz)
# ax.axes.set_xlim3d(left=-.6, right=.6) 
# ax.axes.set_ylim3d(bottom=-.6, top=.6) 
# ax.axes.set_zlim3d(bottom=0, top=.9)
# plt.show()