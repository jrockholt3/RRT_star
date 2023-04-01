import numpy as np

dt = 0.016 # time step
t_limit = 4 # time limit in seconds
thres = np.array([.03, .03, .03]) # joint error threshold
vel_thres = thres # joint velocity error threshold for stopping
# weighting different parts of the reward function
prox_thres = .05 # proximity threshold - 5 cm
min_prox = 0.1
vel_prox = 0.5

# Controller gains
tau_max = 30 #J*rad/s^2, J = 1
damping = tau_max*.5
P = 500
D = 0
P_v = 10.0
D_v = 5.0
jnt_vel_max = np.pi # rad/s
j_max = tau_max/dt # maximum allowable jerk on the joints