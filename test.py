from trajectory import Trajectory
import numpy as np
from env_config import dt

th1 = np.zeros(3)
w1 = np.ones(3)*np.pi/4
tau = np.random.rand(3) * np.pi

traj = []
traj.append((0,th1,w1))
for i in range(0,5):
    w2 = tau*dt + w1
    th2 = tau*dt**2/2 + w1*dt + th1

traj.append((5,th2,w2))
traj.reverse()

tau_list = Trajectory(traj)
print('tau_list', tau_list)
print('tau', tau)
print('traj', traj)
print('error',np.round(tau-tau_list[0], 4))

