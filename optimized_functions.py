from numba import njit, int32, float64
import numpy as np
from env_config import dt, t_limit, thres, vel_thres, jnt_vel_max, min_prox, vel_prox, prox_thres, damping as Z
from Robot3D import robot_3link
robot = robot_3link()
jnt_max = robot.jnt_max
jnt_min = robot.jnt_min

@njit((float64[:])(float64, float64, float64, float64), nogil=True)
def T_ji(thj, aij, lij, Sj):
    return np.array([[np.cos(thj),                        -np.sin(thj),            0,              lij],
                     [np.sin(thj)*np.cos(aij), np.cos(thj)*np.cos(aij), -np.sin(aij), -np.sin(aij)*Sj],
                     [np.sin(thj)*np.sin(aij), np.cos(thj)*np.sin(aij),  np.cos(aij),  np.cos(aij)*Sj],
                     [                    0.0,                     0.0,          0.0,             1.0]])

@njit((float64[:])(float64),nogil=True)
def T_1F(ph, S):
    return np.array([[np.cos(ph), -np.sin(ph), 0, 0],
                     [np.sin(ph),  np.cos(ph), 0, 0],
                     [    0,      0, 1, S], 
                     [    0,      0, 0, 1]])

@njit((float64[:])(float64[:]),nogil=True)
def T_inverse(T):
    R = T[0:3,0:3].T
#     print(R)
    shift = T[0:3,3]
#     print('shift ' + str(shift))
    shift = -R@shift
#     print('-shift ' + str(shift))
    new_T = np.zeros((4,4))
    for i in range(0,3):
        for j in range(0,3):
            new_T[i,j] = R[i,j]
    for i in range(0,3):
        new_T[i,3] = shift[i]
    new_T[3,3] = 1

@njit((float64[:])(float64[:]),nogil=True)
def T_inverse(T):
    R = T[0:3,0:3].T
#     print(R)
    shift = T[0:3,3]
#     print('shift ' + str(shift))
    shift = -R@shift
#     print('-shift ' + str(shift))
    new_T = np.zeros((4,4))
    for i in range(0,3):
        for j in range(0,3):
            new_T[i,j] = R[i,j]
    for i in range(0,3):
        new_T[i,3] = shift[i]
    new_T[3,3] = 1

@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
def asb_link2(obj_pos, th, a, l, S):
    _2toF = T_1F(th[0], S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[1],a[0],l[0],S[1])
    inv_T = T_inverse(_2toF)
    vec = np.array([obj_pos[0],obj_pos[1],obj_pos[2],1])
    return inv_T@vec

@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
def asb_link3(obj_pos, th, a, l, S):
    _3toF = T_1F(th[0], S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])
    inv_T = T_inverse(_3toF)
    vec = np.array([obj_pos[0],obj_pos[1],obj_pos[2],1])
    return inv_T@vec

@njit((float64)(float64[:],float64[:],float64[:],float64[:],float64[:]))
def proximity(obj_pos, th, a, l, S):
    obj_i = obj_pos[:,i]
    asb_l2 = asb_link2(obj_i,th,a,l,S)
    asb_l3 = asb_link3(obj_i,th,a,l,S)
    obj_arr = np.array([asb_l2[0:3], asb_l3[0:3]])
    min_prox = np.inf
    for i in range(0,2):
        obj_i = obj_arr[:,i]
        if obj_i[0] <= 0.0:
            prox = np.linalg.norm(obj_i)
        elif obj_i[0] >= l[i+1]:
            prox = np.linalg.norm(obj_i)
        else:
            prox = np.linalg.norm(obj_i[1:3])
        if min_prox > prox:
            min_prox = prox

    return min_prox

@njit((float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
def njit_forward(th, a, l, S, P_3):
    return T_1F(th[0],S[0])@T_ji(th[1],a[0],l[0],S[1])@T_ji(th[2],a[1],l[1],S[2])@P_3    

@njit((float64[:])(float64),nogil=True)
def calc_clip_vel(prox):
    return jnt_vel_max * (1 - np.exp(-(2/3)*(prox-min_prox)/(vel_prox-min_prox)**2))

@njit((float64[:],float64[:])(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:]),nogil=True)
def nxt_state(obj_pos, th, w, tau, a, l, S):
    '''
    input current state of the robot
        th, w
    input action 
    check proximity
        if prox violation - stop the robot return new paused state,
        else update state with action
    with updated state check
        safety violations
            if safety violations, update robot position + vel to safe position
        check termination conditions
    calculate reward
    return new state, new_th, new_w
    '''
    prox = -np.inf
    prox_arr = np.zeros(obj_pos.shape[1])
    for i in range(0,obj_pos.shape[1]):
        prox_arr[i] = proximity(obj_pos[:,i], th, a, l, S)
    prox = np.min(prox_arr)

    if prox <= vel_prox:
        vel_clip = calc_clip_vel(prox)
    else:
        vel_clip = jnt_vel_max

    paused = False
    if prox < min_prox:
        paused = True
    
    if not paused: 
        nxt_w = (tau - Z*w)*dt + w
        nxt_th = (tau - Z*w)*dt**2/2 + w*dt + th

        nxt_w = np.clip(nxt_w,-vel_clip,vel_clip)
    else:
        nxt_w = np.zeros_like(w)
        nxt_th = th

    # check joint limits 
    if np.any(nxt_th >= jnt_max) or np.any(nxt_th <= jnt_min):
        nxt_w[nxt_th >= jnt_max] = 0
        nxt_w[nxt_th <= jnt_min] = 0
        nxt_th[nxt_th >= jnt_max] = jnt_max[nxt_th >= jnt_max]
        nxt_th[nxt_th <= jnt_min] = jnt_min[nxt_th <= jnt_min]

    return nxt_th, nxt_w


