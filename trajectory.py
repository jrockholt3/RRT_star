import numpy as np
from env_config import dt, t_limit

def Trajectory(traj):
    '''
    traj: list of tups with each entry (t, th, w) being a saved
    wave point along a path
    returns: an array of tau's from t=0 to t=t_final
    '''

    def tau(t, t1, a, b):
        return (6*a)*(t - t1) + 2*b

    def th(t, t1, a, b, c, d):
        t_ = t - t1
        return (a*t_**3)

    (t1,th1,w1, targ) = traj.pop()
    t = t1
    tau_list = []
    while len(traj) > 0:
        # th(t) = at**3 + bt**2 + c*t + d
        (t2, th2, w2, targ) = traj.pop()
        
        d = th2
        c = w2
        # Ax = b
        # x = (a; b)
        t_ = dt*(t2 - t1)
        A = np.array([[t_**3, t_**2],
                      [3*t_**2, 2*t_]])
        b = np.array([[th2-w1*t_-th1], [w2-w1]])
        a_vec = np.zeros(b.shape[2])
        b_vec = np.zeros(b.shape[2])
        for i in range(b.shape[2]):
            x_i,R,rank,S = np.linalg.lstsq(A,b[:,:,i])
            a_vec[i] = x_i[0]
            b_vec[i] = x_i[1]

        while t < t2 and t < t_limit/dt:
            tau_list.append((tau(t*dt, t1*dt, a_vec, b_vec),targ))
            t += 1
    
        t1 = t2
        t = t1
        th1 = th2
        w1 = w2
    
    return tau_list
