import torch 
import MinkowskiEngine as ME
import numpy as np
import Robot_Env
from Robot_Env import RobotEnv
from rtree import index
import uuid
    
def act_preprocessing(state,single_value=False,device='cuda'):
    if single_value:
        coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
        jnt_pos = state[2]#.clone().detach()
        jnt_pos = torch.tensor(jnt_pos,dtype=torch.double,device=device).view(1,state[2].shape[0])
        jnt_goal = state[3]
        jnt_goal = torch.tensor(jnt_goal,dtype=torch.double,device=device).view(1,jnt_goal.shape[0])
    else:
        coords,feats = ME.utils.sparse_collate(state[0],state[1])
        jnt_pos = state[2]#.clone().detach()
        jnt_pos = torch.tensor(jnt_pos,dtype=torch.double,device=device)
        jnt_goal = state[3]
        jnt_goal = torch.tensor(jnt_goal,dtype=torch.double,device=device)

    x = ME.SparseTensor(coordinates=coords, features=feats.double(),device=device)
    return x, jnt_pos, jnt_goal

def crit_preprocessing(state, action, single_value=False, device='cuda'):
    if single_value:
        coords,feats = ME.utils.sparse_collate([state[0]],[state[1]])
        jnt_pos = state[2]#.clone().detach()
        jnt_pos = torch.tensor(jnt_pos,dtype=torch.double,device=device).view(1,state[2].shape[0])
        jnt_goal = state[3]
        jnt_goal = torch.tensor(jnt_goal,dtype=torch.double,device=device).view(1,jnt_goal.shape[0])
        a = torch.tensor(action,dtype=torch.double,device=device).view(1,action.shape[0])
    else:
        coords,feats = ME.utils.sparse_collate(state[0],state[1])
        jnt_pos = state[2]#.clone().detach()
        jnt_pos = torch.tensor(jnt_pos,dtype=torch.double,device=device)
        jnt_goal = state[3]
        jnt_goal = torch.tensor(jnt_goal,dtype=torch.double,device=device)
        a = torch.tensor(action,dtype=torch.double,device=device)

    x = ME.SparseTensor(coordinates=coords, features=feats.double(),device=device)
    return x, jnt_pos, jnt_goal, a

def create_stack(state:tuple):
    coord_list = []
    feat_list = []
    c_arr = state[0]
    for j in range(6):
        c_arr[:,0] = j
        coord_list.append(c_arr.copy())
        feat_list.append(state[1])
    return coord_list, feat_list

def stack_arrays(coord_list:list, feat_list:list, new_state:tuple):
    coord_list.insert(0,new_state[0])
    feat_list.insert(0,new_state[1])
    if len(coord_list) > 6:
        coord_list.pop()
        feat_list.pop()
    new_list = []
    for j,c in enumerate(coord_list):
        c[:,0] = j
        new_list.append(c.copy())
    coord_list = new_list.copy()

    return coord_list, feat_list


# def gen_obstacles(env:RobotEnv, obs_index:index.Index):
#     '''
#     have to generate the min and max corners of obstacles at 
#     each time step
#     '''
#     objs = env.objs.copy()
#     dt = Robot_Env.dt
#     t_limit = Robot_Env.t_limit
#     min_prox = Robot_Env.min_prox
#     obs = []

#     t = 0
#     time_steps = int(np.ceil(t_limit/dt))
#     while t <= time_steps:
#         for o in objs:
#             center = o.curr_pos
#             x,y,z = center[0],center[1],center[2]
#             obs_i = (t, x-min_prox, y-min_prox, z-min_prox, t, x+min_prox, y+min_prox, z+min_prox)
#             obs_index.add(uuid.uuid4(), obs_i)
#             obs.append(obs_i)
#             o.step()
#         t += 1

#     return obs

        