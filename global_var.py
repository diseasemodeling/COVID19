import pandas as pd
import datetime
import numpy as np
import os
from progress.bar import IncrementalBar

def setup_global_variables(state, inv_dt1, T_max_, lead_time_, time_unit_, beta_user_defined_, path):
    ##### do not change ######
    global tot_risk
    tot_risk = 2

    global tot_age
    tot_age = 101
    ##### do not change ######
    
    ##### user defined input #######
    global enter_state
    enter_state = state

    global T_max
    T_max = T_max_

    global inv_dt
    inv_dt = inv_dt1

    global dt
    dt = 1/inv_dt

    global lead_time
    lead_time = lead_time_ 

    global time_unit
    time_unit = time_unit_ 

    global beta_user_defined
    beta_user_defined = beta_user_defined_

    ##### user defined input #######

    ##### read simulation input #######
    sim_result = read_sim_input(state = enter_state, cwd = path)

    final_simul_start_date = sim_result[0]

    sd_start_date_state = sim_result[1]

    begin_simul_rl_date = sim_result[2]

    global days_of_simul_pre_sd
    days_of_simul_pre_sd = int(sim_result[3])

    global days_of_simul_post_sd
    days_of_simul_post_sd = int(sim_result[4])

    global dry_run_end_diag
    dry_run_end_diag = int(sim_result[5])

    global symp_hospitalization_v
    symp_hospitalization_v = sim_result[6]

    global percent_dead_recover_days_v
    percent_dead_recover_days_v = sim_result[7]

    global pop_dist_v
    pop_dist_v = sim_result[8]

    global input_list_const_v # dataframe
    input_list_const_v = sim_result[9]
    
    global Q
    Q = sim_result[10]

    beta_vals = sim_result[11]
    global beta_before_sd 
    beta_before_sd  = beta_vals[0]

    global beta_after_sd
    beta_after_sd = beta_vals[1]

    global hosp_scale
    hosp_scale = beta_vals[2]

    global dead_scale
    dead_scale  = beta_vals[3]

    global rates_indices
    rates_indices = reading_indices(Q)

    global diag_indices
    diag_indices = diag_indices_loc(Q)

    ##### read RL input #######
    rl_result = read_RL_inputs(state = enter_state, start_sim_date = final_simul_start_date, cwd = path)

    global VSL
    VSL = rl_result[0]

    global lab_for
    lab_for = rl_result[1]

    global K_val
    K_val = rl_result[2]

    global A_val
    A_val = rl_result[3]

    global duration_unemployment
    duration_unemployment = rl_result[4]

    global init_unemploy
    init_unemploy = rl_result[5]

    global md_salary
    md_salary = rl_result[6]/8 *(40/7)

    global test
    test = rl_result[7]
    
    prog_d = days_of_simul_pre_sd + days_of_simul_post_sd + T_max
    global prog_bar      # progress bar to show how long until the end of running code
    prog_bar = IncrementalBar('Code In Processing: \t', max = prog_d + 1)

    global actual_data 
    global acutal_unemp
    actual_data, acutal_unemp = read_actual_data(state = enter_state, cwd = path)

    print('\n')

    global decision
    global decision_week
    decision_week = read_decisions(cwd = path)


# Function to read actual data
# Input parameters:
# state: the State you want to model
# cwd: the current working directory of running code
def read_actual_data(state, cwd):
    excel1 = os.path.join(cwd,'data/actual_valid_data.xlsx')
    df = pd.read_excel(excel1, sheet_name='Sheet1')
    df_state = df[df['state'] == state]
    # df_state_1 = df_state[['date', 'positive','death', 'hospitalized']]
    df_state_1 = df_state.loc[:, ('date', 'positive','death', 'hospitalized')]
    df_state_1['date'] = pd.to_datetime(df_state_1['date'], format='%Y%m%d', errors='coerce')
    df_state_1.rename(columns= {'positive':'actual cumulative diagnosis', 'death':'actual cumulative deaths',\
                      'hospitalized': 'actual cumulative hospitalized'}, inplace=True)

    excel2 = os.path.join(cwd, 'data/RL_input.xlsx')
    df2 = pd.read_excel(excel2, sheet_name = 'actual_unemploy')
    df2_state = df2.loc[:, ('Date', state)]
    df2_state.rename(columns = {state: 'Actual unemployment rate'}, inplace = True)
    df2_state['Date'] = pd.to_datetime(df2_state['Date'], format='%Y%m%d', errors='coerce')

    return df_state_1, df2_state
# Return 2 values
# [0] = df_state_1: a dataframe with actual cumulative diagnosis, deaths, hospitalizations by date
# [1] = df2_state: a dataframe with actual unemployment rate by date 


# Function to read simulation related parameters
def read_sim_input(state, cwd):
    excel1= os.path.join(cwd,'data/COVID_input_parameters.xlsx')
    excel2 = os.path.join(cwd, 'data/pop_dist.xlsx')
    excel3 = os.path.join(cwd, 'data/actual_valid_data.xlsx')
    excel4 = os.path.join(cwd,'data/states_beta.xlsx')

    # read blank Q-matrix 
    q_mat_blank = pd.read_excel(excel1, sheet_name = 'q-mat_blank') 
    q_mat_blank_v = q_mat_blank.values

    # read input paramteres 
    input_list_const = pd.read_excel(excel1, sheet_name = 'input_list_const', index_col = 0) 

    # read age related hospitalization probabilities
    symp_hospitalization = pd.read_excel(excel1, sheet_name='symp_hospitalization') 
    symp_hospitalization_v = symp_hospitalization.values

    # read age and gender related death and recovery probabilities and time from day of hospitalization
    percent_dead_recover_days = pd.read_excel(excel1, sheet_name='percent_dead_recover_days') 
    percent_dead_recover_days_v = percent_dead_recover_days.values

    # read date of social distancing 
    sd_date = pd.read_excel(excel1, sheet_name='sd_date') 
    sd_date_v = sd_date.values 

    # read population distribution of the State
    pop_dist = pd.read_excel(excel2, sheet_name = state) 
    pop_dist_v = pop_dist.values

    # read actual data 
    raw_valid_data = pd.read_excel(excel3, sheet_name='Sheet1')
    raw_valid_data_v = raw_valid_data.values
    state_index = np.where(raw_valid_data_v == state)
    valid_data_v = raw_valid_data_v[state_index[0]]

    # The beginning date where the number of infections exceed 0
    final_simul_start_date = valid_data_v[max(np.where(valid_data_v[:,2] > 0)[0]), 0]
    final_simul_start_date = str(final_simul_start_date)
    d1 = datetime.datetime.strptime(final_simul_start_date, '%Y%m%d')

    # The ending date where the number of infections exceed 0
    final_simul_end_date = valid_data_v[min(np.where(valid_data_v[:,2] > 0)[0]), 0]
    final_simul_end_date = str(final_simul_end_date)
    d2 = datetime.datetime.strptime(final_simul_end_date, '%Y%m%d')

    # Min number of infections that need to be diagnosed for dry run to end
    dry_run_end_diag = valid_data_v[max(np.where(valid_data_v[:,2] > 0)[0]), 2] 

    # The date where social distancing begins in the state
    sd_start_date_state = str(sd_date_v[np.where(sd_date_v == state)[0][0],1])
    sd_start_date_state = str(datetime.datetime.strptime(sd_start_date_state, "%Y-%m-%d %H:%M:%S"))
    d4= datetime.datetime.strptime(sd_start_date_state, '%Y-%m-%d %H:%M:%S')

    # The days before social distancing was introduced
    days_of_simul_pre_sd = (abs(d4 - d1).days)

    # The days after social distancing was introduced 
    days_of_simul_post_sd = (abs(d4 - d2).days)+1

    # The days when decision making begins
    begin_simul_rl_date = str(valid_data_v[min(np.where(valid_data_v[:,2] > 0)[0]), 0])

    # beta for the State
    states_betas = pd.read_excel(excel4, sheet_name = 'Sheet1', index_col = 0)
    beta_v = states_betas.loc[state]

    return (final_simul_start_date, sd_start_date_state, begin_simul_rl_date, days_of_simul_pre_sd, 
            days_of_simul_post_sd, dry_run_end_diag, symp_hospitalization_v, percent_dead_recover_days_v, 
            pop_dist_v, input_list_const, q_mat_blank_v, beta_v)

    
# Returns 12 values 
# [0] = final_simul_start_date - the day when simulation begins, format YMD
# [1] = sd_start_date_state - the day when social distancing begins, format YMD H:M:S
# [2] = begin_simul_rl_date - the day RL simulation is begun, format YMD
# [3] = days_of_simul_pre_sd - difference between [0] and [1] 
# [4] = days_of_simul_post_sd - difference between [1] and [2]
# [5] = dry_run_end_diag - number of diagnoses at end of dry run
# [6] = symp_hospitalization_v - the hospitalization data 
# [7] = percent_dead_recover_days_v - recovery and death data
# [8] = pop_dist_v - inital population distribution of susceptible by age and risk group
# [9] = input_list_const_v - list of input parameters
# [10] = q_mat_blank_v - an np array of size 10x10 with values 1 where rates are >0
# [11] = beta_v - beta value for the State (max and min), hospitalization scale and death scale
 

# Function to read and extract indices of the q mat where value is = 1
# Input parameters for this function
# NULL
def reading_indices(Q):
    rate_indices = np.where(Q == 1) 
    list_rate = list(zip(rate_indices[0], rate_indices[1]))
    return list_rate
# Returns 1 value
# A list of length 16, represents the compartment flow; e.g. (0,1) represents 0 -> 1


# Function to extract indices of the diagonal of the q mat where value
# Input parameters for this function
# NULL
def diag_indices_loc(Q):
    mat_size = Q.shape
    diag_index = np.diag_indices(mat_size[0], mat_size[1])
    diag_index_fin = list(zip(diag_index[0], diag_index[1]))
    
    return diag_index_fin
# Returns 1 value
# An np array of size 10x1. Here we have a 10x10 q mat so we have 10 diagonal values. 
# e.g. (0,0) represents 0 -> 0, (1,1) represents 1 -> 1


# Function to read date related to simulation
# Input parameters:
# state: the State you want to model
# cwd: the current working directory of running code
def read_date(state, cwd):
    excel1 = os.path.join(cwd,'data/actual_valid_data.xlsx')
    raw_valid_data = pd.read_excel(excel1, sheet_name='Sheet1')
    raw_valid_data_v = raw_valid_data.values
    state_index = np.where(raw_valid_data_v==state)
    valid_data_v = raw_valid_data_v[state_index[0]]
    final_simul_start_date = valid_data_v[max(np.where(valid_data_v[:,2] > 0)[0]), 0]
    final_simul_start_date = str(final_simul_start_date)
    d1 = final_simul_start_date
    begin_simul_rl_date = str(valid_data_v[min(np.where(valid_data_v[:,2] > 0)[0]), 0])
    excel2= os.path.join(cwd,'data/COVID_input_parameters.xlsx')
    sd_date = pd.read_excel(excel2, sheet_name='sd_date') 
    sd_date_v = sd_date.values 
    sd_start_date_state = str(sd_date_v[np.where(sd_date_v == state)[0][0],1])
    d2 = sd_start_date_state
    d3 = begin_simul_rl_date

    return d1, d2, d3 


# Function to read RL input
# Input paramters:
# state - State you want to model
# start_sim_date -  start date of simulation 
# cwd - the current working directy for running code
def read_RL_inputs(state, start_sim_date, cwd):
    # read data
    excel = os.path.join(cwd, 'data/RL_input.xlsx')
    df = pd.ExcelFile(excel)
    # read VSL
    VSL1 = df.parse(sheet_name='VSL_mod') 
    VSL2 = VSL1.to_numpy()
    VSL3 = np.transpose(VSL2)
    VSL = VSL3[:][1]

    # read labor force participation rate
    lab_for_v = df.parse(sheet_name='labor_for')
    val = lab_for_v[state].values[0]/100

    # read maximum and minimum unemployment rate
    cof_unemploy = df.parse(sheet_name='unemploy_cof', index_col = 0) 
    K_val = cof_unemploy.loc['max', state]/100  
    A_val = cof_unemploy.loc['min', state]/100
    
    # read initial unemployment rate
    day = pd.Timestamp(start_sim_date)
    df2 = df.parse(sheet_name='actual_unemploy')
    df_state = df2.loc[:, ('Date', state)]
    df_state['Date'] = pd.to_datetime(df_state['Date'], format='%Y%m%d', errors='coerce')
    df_state = df_state[(df_state['Date']>day)]
    df_state = df_state.reset_index(drop=True)
    init_unemploy = df_state.loc[0, state]/100
    

    # read duration from start of social distancing to the maximum of unemployment rate
    dur = df.parse(sheet_name = 'duration_unemploy', index_col=0)
    dur = pd.to_datetime(dur[state], format='%Y%m%d', errors='coerce')
    max_date = dur.loc['max_date']
    sd_date = dur.loc['sd_date']
    duration_unemployment = abs(max_date - sd_date).days
    
    # read median wage and cost of testing by type
    others = df.parse(sheet_name='others') 
    others_list = others.values.tolist()
    md_salary = others_list[0][0]
    test = others_list[0][1:]
  
    return  VSL, val, K_val, A_val, duration_unemployment, init_unemploy, md_salary, test

# Returns 8 values
# [0] = VSL - A np array of size 101x1 which is the value of statistical life by age 
# [1] = val - A float value for labor force participation rate, assumed constant
# [2] = K_val - A float value for maximum unemployment rate 
# [3] = A_val - A float value for minimum unemployment rate 
# [4] = duration_unemployment - An integer value for the duration between the start of social distancing to the maximum
# [5] = init_unemployment - A float value for the initial unemployment rate once the simulation starts
# [6] = md_salary - median wage in a day 
# [7] = test - A list of size 1x3 which is the cost of symptom-based testing, contact tracing and universal testing


def read_decisions(cwd):
    dir_ = os.path.join(cwd, 'data/decision_making.csv')
    df = pd.read_csv(dir_) 
    df_v = df.to_numpy(dtype = float)
    x = np.copy(df_v[:,1:])
    x[:,0] = 1 
    return x

def get_decisions(STR1, STR2, STR3, T):
    W = change_W(decision_week,STR1,STR2,STR3)
    return format_year(W,T)

def format_year(W, total_day = 365):
    D = np.zeros((total_day, W.shape[1]))
    for i in range(W.shape[0]):
        for j in range(7):
            D[7*i+j] = W[i]
            if 7 * i + j == total_day -1:
                return D
    D[-1] = W[-1]
    return D        

def change_W(W,STR1,STR2,STR3):
    W_m = np.copy(W)
    _change_W(W_m,STR1,0)
    _change_W(W_m,STR2,1)
    _change_W(W_m,STR3,2)
    return W_m

def _change_W(W,STR,column):
    if STR =='':
        return None
    
    input_dic = modify_input(STR)
    for key in input_dic.keys():
        for i in range(key[0],key[1]):
            W[i,column] = input_dic[key]
    
def modify_input(STR):
    l = STR.split(',')
    length = len(l) // 3
    if len(l) % 3!= 0:
        raise ValueError('input shou be a mutiple of 3')
        
    dic = {}
    for i in range(length):
        dic[(int(l[3*i]) -1 ,int(l[3*i + 1]))] = float(l[3*i + 2])
    return dic   

    