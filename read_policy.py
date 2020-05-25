import pandas as pd
from math import isnan
import numpy as np
import os


# input for start day to end day should be small to large, start from 1
# input date range should be less than T_max

def read_policy(T_max, policy_range, init_policy, path):

    # read template
    df = pd.read_excel(path, header =0)
    params = df.values.shape[1]//3 # number of policy parameters

    # initialize policy in the form of that we want
    policy = np.zeros((T_max, params))


    l = []
    for k in range(params):
        l.append([])
    
    for i in range(df.values.shape[0]):
        for j in range(params):
            if not isnan(df.values[i][params*j]):
                start_day = int(df.values[i][params*j])-1
                end_day = int(df.values[i][params*j+1])
                val = df.values[i][params*j+2]
                if val >= policy_range[j][0] and val <= policy_range[j][1]:
                    l[j].append((start_day, end_day, val))
                else:
                    raise Exception("Sorry, your decision choices goes beyond the designed range")

    for j in range(params):
        i = 0
        if len(l[j]) != 0:
            for i in range(l[j][0][0]):
                policy[i][j] =  init_policy[j]
            i += 1
            v = l[j]
            for k in range(len(v)):
                if k == 0:
                    pre_value = init_policy[j]
                else:
                    pre_value = v[k -1][2]
                    
                while i < v[k][0]:
                    policy[i][j] = pre_value
                    i += 1
                
                for i in range(v[k][0],v[k][1]):
                    policy[i][j]  = v[k][2]
                i += 1
            pre_value = v[-1][2]
            while i <= T_max - 1:
                policy[i][j] = pre_value
                i += 1
        else:
            policy[:,j] = init_policy[j]
                 
    return policy
        
    