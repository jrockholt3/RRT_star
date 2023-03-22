import Robot_Env
from Robot_Env import dt, t_limit
import numpy as np
from rtree import index
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

env = Robot_Env.RobotEnv()

# print(np.ceil(t_limit/dt)*3)

# p = index.Property()
# p.dimension = 4
# obs = index.Index(interleaved=True, properties=p)
# obs_list = gen_obstacles(env,obs)

# th = np.random.uniform(env.robot.jnt_min, env.robot.jnt_max)
# env.robot.set_pose(th)
# while env.robot.check_safety():
#     th = np.random.uniform(env.robot.jnt_min, env.robot.jnt_max)
#     env.robot.set_pose(th)

# ptns = env.robot.forward()
# print(ptns)
# res = .01
# n = 10
# t = 30
# arr = []
# for i in range(1,ptns.shape[1]-1):
#     x1 = ptns[0:3,i]
#     x2 = ptns[0:3,i+1]
#     u = (x2 - x1)
#     for i in range(1,n+1):
#         new = u*i/n + x1
#         tup = (t, new[0], new[1], new[2])
#         arr.append(tup)

# # for x_ in arr:
#     print(obs.count(x_)==0)
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

from Robot_Env import dt, tau_max, j_max as J, jnt_vel_max

# T = 4*dt
# print('current search radius is', T*jnt_vel_max)

# r = -(J/6)*(T/2)**3 + (J*T/4)*(T/4)**2 + (J*T/12)*(T/2) + J*T**3/12

# print('Jmx search radius is', r)

env = Robot_Env.RobotEnv(has_objects=False)

done = False
th = []
w = []
tau = []
start = np.zeros(3)
goal = start + np.pi/4
g = []
env = Robot_Env.RobotEnv(has_objects=True, start=start, goal=goal)
while not done:
    _,_,done,info = env.step(np.zeros(3),eval=True, use_VControl=True, w=np.ones(3)*Robot_Env.jnt_vel_max/6)
    tau.append(info['tau'])
    th.append(env.robot.pos)
    w.append(env.robot.jnt_vel)
    g.append(goal)


fig1 = plt.figure()
plt.plot(np.arange(len(w)), w)
plt.title('Velocity')
fig2 = plt.figure()
plt.plot(np.arange(len(th)), th)
plt.plot(np.arange(len(g)), g)
plt.title('Ang_Pos')
fig3 = plt.figure()
plt.plot(np.arange(len(tau)), tau)
plt.title('Tau')
plt.show()