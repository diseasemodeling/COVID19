# -*- coding: utf-8 -*-

import pandas as pd
from math import isnan
import numpy as np
"""
global default_values 
default_values = [0.1,0.2,0.3]
# input should from samll values to larger values
# input date should be less than Tmax
# day should strat from 1

def read_policy(path,Tmax):
    df = pd.read_excel(path, header =0)
    
    params = df.values.shape[1]//3
    l = []
    for k in range(params):
        l.append([])
    
    for i in range(df.values.shape[0]):
        for j in range(params):
        # policy 1
            if not isnan(df.values[i][params*j]):
                l[j].append((int(df.values[i][params*j])-1,int(df.values[i][params*j+1]),df.values[i][params*j+2]))
    
    policy = np.zeros((Tmax,params))
    
    for j in range(params):
        i = 0
        if len(l[j]) != 0:
            for i in range(l[j][0][0]):
                policy[i][j] =  default_values[j]
            i += 1
            v = l[j]
            for k in range(len(v)):
                if k == 0:
                    pre_value = default_values[j]
                else:
                    pre_value = v[k -1][2]
                    
                while i < v[k][0]:
                    policy[i][j] = pre_value
                    i += 1
                
                for i in range(v[k][0],v[k][1]):
                    policy[i][j]  = v[k][2]
                i += 1
            pre_value = v[-1][2]
            while i <= Tmax - 1:
                policy[i][j] = pre_value
                i += 1
        else:
            policy[:,j] = default_values[j]
                

        
    return policy"""

def read_policy(path):
    df = pd.read_excel(path, header =0)
    l = []
    params = 3
    for i in range(params):
        l.append([])
    
        
    for j in range(params):
        for i in range(1,df.values.shape[0]):
        
        # policy 1
            if not isnan(df.values[i][3*j+1]):
                l[j].append((df.values[i][3*j+1]-1,df.values[i][3*j+2]))
            
            else:
                break
            

    T_max = l[0][-1][0] +1
    policy = np.zeros((T_max,params))
    for j in range(params):
        for k in range(len(l[j])-1):
            policy[l[j][k][0]:l[j][k+1][0],j] = l[j][k][1]
        
        policy[T_max-1,j] = l[j][k+1][1]
    return policy
            
"""if __name__ =='__main__':
    path = 'policy_example.xlsx'
    
    policy = read_policy(path)"""