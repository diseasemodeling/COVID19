import numpy as np 
import pandas as pd
import os
import datetime

import global_var as gv
import outputs as op
import pdb


class CovidModel():
    def __init__(self, path):
        super(CovidModel, self).__init__()

        self.beta_max = gv.beta_before_sd  # max transmission rate
        self.beta_min = gv.beta_after_sd   # min transmission rate
        """if gv.beta_user_defined != 0:      # user-defined transmission rate
            self.beta = gv.beta_user_defined
        else: 
            self.beta = self.beta_max"""
        
        self.enter_state = gv.enter_state  # two letter abbreviation of the state you want to model

        self.tot_risk = gv.tot_risk        # total risk group: female, male
        self.tot_age = gv.tot_age          # total age group: 101, by age

        self.a_sd_range = [0, 1 - (self.beta_min/self.beta_max)]  # social distancing range
        
        self.T = gv.T_max
        self.inv_dt = gv.inv_dt 
        self.dt = 1/self.inv_dt
        
        """ ##### this one would be incorporated in the later stages of modeling, not used now
        self.lead_time = gv.lead_time          # the time period before the action takes effect"""  

        ### simulation related variables; won't change during the simulation 
        self.Q = gv.Q                                                      # Q-matrix
        self.num_state = self.Q.shape[0]
        self.rates_indices = gv.rates_indices                              # rate matrix
        self.diag_indices = gv.diag_indices                                # diagonal matrix
        self.symp_hospitalization = gv.symp_hospitalization_v            
        self.percent_dead_recover_days = gv.percent_dead_recover_days_v  
        self.input_list_const = gv.input_list_const_v                      # input parameters
        self.init_pop_dist = gv.pop_dist_v                                 # initial population distribution 
        self.tot_pop = np.sum(self.init_pop_dist)                          # total number of population
        self.dry_run_end_diag = gv.dry_run_end_diag                        # after dry run, total number of diagnosis should match with data
        self.days_of_simul_pre_sd = gv.days_of_simul_pre_sd
        self.days_of_simul_post_sd = gv.days_of_simul_post_sd  

        # simulation time period since dry run
        self.T_total = self.inv_dt * (self.T + self.days_of_simul_pre_sd + self.days_of_simul_post_sd) # simulation time period from dry run
      
        ### read value, ub or lb for each items
        self.l_days =  self.input_list_const.loc['l_days', 'value']
        self.prop_asymp = self.input_list_const.loc['prop_asymp', 'value']
        self.incub_days = self.input_list_const.loc['incub_days', 'value']
        self.a_b = self.input_list_const.loc['a_b', 'value']
        self.ir_days = self.input_list_const.loc['ir_days', 'value']
        self.qih_days = self.input_list_const.loc['qih_days', 'value']
        self.qir_days = self.input_list_const.loc['qir_days', 'value']
        self.hosp_scale = 1   # hospitalization scale
        self.dead_scale = 1   # death scale

        # rl related parameters
        self.lab_for = gv.lab_for
        self.VSL = gv.VSL
        self.md_salary = gv.md_salary/self.inv_dt
        self.K_val = gv.K_val
        self.A_val = gv.A_val
        self.h_val = gv.h_val
        self.rl_counter = 0                   # every time, when calculate immediate reward, counter will increment by 1
        self.cost_tst = gv.test  
        self.op_ob = op.output_var(int(self.T_total/self.inv_dt) + 2, state = self.enter_state, cwd = path)

        self.reset_sim()                       # reset the simulation 
        self.reset_rl()                        # reset rl 
        
    def step(self, action_t, beta):
        # self.policy[self.t] = action_t  # record action at t time step
        self.simulation_base(action_t, beta)
        self.calc_imm_reward(action_t)
        self.output_result() 
                                 
    def output_result(self):
        if self.t % self.inv_dt == 0: 
            self.d += 1
            self.op_ob.time_step[self.d] = self.d     # timestep (day)

            #### if plot for one day
            indx_l = self.t - self.inv_dt + 1 # = self.t
            indx_u = self.t + 1  # = self.t + 1
            self.op_ob.num_inf_plot[self.d] = np.sum(self.num_diag[indx_l: indx_u])        # new infected at timestep t
            self.op_ob.num_hosp_plot[self.d] = np.sum(self.num_hosp[indx_l: indx_u])       # new hosp at timestep t
            self.op_ob.num_dead_plot[self.d] = np.sum(self.num_dead[indx_l: indx_u])       # new dead at timestep t
            self.op_ob.num_base[self.d] = np.sum(self.num_base_test[indx_l: indx_u]) 
            self.op_ob.num_uni[self.d] = np.sum(self.num_uni_test[indx_l: indx_u]) 
            self.op_ob.num_trac[self.d] = np.sum(self.num_trac_test[indx_l: indx_u]) 
            self.op_ob.num_hop_tst[self.d] =  np.sum(self.num_hosp_test[indx_l: indx_u]) 
            self.op_ob.cumulative_inf[self.d] =  self.tot_num_diag[self.t]       # cumulative infections from start of sim to timestep t
            self.op_ob.cumulative_hosp[self.d] = self.tot_num_hosp[self.t]       # cumulative hospitalizations from start of sim to timestep t
            self.op_ob.cumulative_dead[self.d] = self.tot_num_dead[self.t]       # cumulative dead from start of sim to timestep t 
            
            if self.rl_counter > 0:
                self.op_ob.VSL_plot[self.d] = -1 * (np.sum(self.Final_VSL[indx_l: indx_u]))    # VSL at timestep t
                self.op_ob.SAL_plot[self.d] = -1 * (np.sum(self.Final_SAL[indx_l: indx_u]))    # SAL at timestep t        
                self.op_ob.unemployment[self.d] = self.rate_unemploy[self.t]
                self.op_ob.univ_test_cost[self.d] = -1 * (np.sum(self.cost_test_u[indx_l: indx_u])) 
                self.op_ob.trac_test_cost[self.d] = -1 * (np.sum(self.cost_test_c[indx_l: indx_u])) 
                self.op_ob.bse_test_cost[self.d] = -1 * (np.sum(self.cost_test_b[indx_l: indx_u])) 
                
    # Function to calculate immediate reward when doing RL
    def calc_imm_reward(self, action_t):
        a_sd = action_t[0]
        a_c = action_t[1] 
        a_u = action_t[2]

        self.rl_counter += 1
    
        # calculate number of unemployment 
        if self.rl_counter % (7 * self.inv_dt) == 0: 
            self.calc_unemployment(a_sd)
        else:
            self.rate_unemploy[self.t] = self.rate_unemploy[self.t - 1]

        tot_alive = self.tot_pop - self.tot_num_dead[self.t]
        
        num_unemploy = tot_alive * self.rate_unemploy[self.t] * self.lab_for/ 10000    # rate converted to percentage

        # calculate total wage loss due to social distancing
        self.Final_SAL[self.t] = num_unemploy * self.md_salary / 1000000  # in million dollars
        
        # calculate total 'value of statistical life' loss due to deaths
        self.Final_VSL[self.t]  = np.sum(np.dot(self.num_dead[self.t] , self.VSL))

        #### calculate cost of testing 
        num_test_u  = np.sum(self.pop_dist_sim[(self.t - 1),:,:,0:4]) # S + L + E + I
        num_test_c = np.sum(self.pop_dist_sim[(self.t - 1),:,:,1:4])  # L + E + I
        num_test_b = np.sum(self.pop_dist_sim[(self.t - 1),:,:,3:4]) # I
        self.cost_test_u[self.t] =  a_u *  num_test_u * self.cost_tst[0]
        self.cost_test_c[self.t] = (1 - a_u)* a_c * num_test_c * self.cost_tst[1]
        self.cost_test_b[self.t] = (1 - (a_u +(1-a_u)*a_c))* self.a_b * num_test_b
        self.Final_TST[self.t] = self.cost_test_u[self.t] + self.cost_test_c[self.t] + self.cost_test_b[self.t]
        
        # calculate immeidate reward 
        self.imm_reward[self.t] = -1 * (self.Final_VSL[self.t]  + self.Final_SAL[self.t] + self.Final_TST[self.t])
     
    # Function to calculate unemployment change
    def calc_unemployment(self, a_sd):
        a_sd_max = self.a_sd_range[1]
    
        prop = a_sd / a_sd_max
        y_p = self.rate_unemploy[self.t-1]

        K = max(prop * self.K_val, y_p)
        A = max(self.A_val, min(prop * self.K_val, y_p))
        u = 1/14
        if y_p == K:
            self.rate_unemploy[self.t] = y_p - u*(K-A)
        else:
            self.rate_unemploy[self.t] = y_p + u*(K-A)
        

    # Function to calculate transition rates (only for the rates that won't change by risk or age)
    def set_rate_array(self, action_t, beta):
        a_sd = action_t[0]
        a_c = action_t[1] 
        a_u = action_t[2]

        self.rate_array[0] = ((1 - a_sd)*(beta * np.sum(self.pop_dist_sim[(self.t - 1),:,:,2:4])))/(np.sum(self.pop_dist_sim[(self.t - 1), :,:,0:9]))
        
        self.rate_array[1] = 1/self.l_days

        self.rate_array[2] = a_u + ((1 - a_u)*a_c)

        self.rate_array[4] = a_u + ((1 - a_u)*a_c)
   
        self.rate_array[6] = self.prop_asymp/(self.incub_days - self.l_days)

        # self.rate_array[7] = a_u + ((1 - a_u)*a_c)+(self.a_b * (1 - (a_u + ((1 - a_u)*a_c))))
        self.rate_array[7] = self.a_b
        
        # self.rate_array[8] = 1/self.ir_days
        self.rate_array[8] = (1 - self.a_b)*(a_u + (1-a_u)*a_c) + 1/self.ir_days

        self.rate_array[9] = 1/self.l_days

        # self.rate_array[10] = (1 - self.prop_asymp)/(self.incub_days - self.l_days)

        # self.rate_array[11] = (self.prop_asymp)/(self.incub_days - self.l_days)


    # Function to perform the base simulation
    # Input parameters for this function
    # action = an np array of size [1x3] with the values output by the RL model (a_sd, a_c, a_u)
    #0	1	2	3	4	5	6	7	8	9 compartments
    #S	L	E	I	Q_L	Q_E	Q_I	H	R	D compartments
    def simulation_base(self, action_t, beta):
        a_sd = action_t[0]
        a_c = action_t[1] 
        a_u = action_t[2]
        
        # Calculate transition rate that won't change during the for loop
        self.set_rate_array(action_t, beta)


        for risk in range(self.tot_risk): # for each risk group i.e, male(0) and female(1)

            for age in range (self.tot_age): # for each age group i.e., from 0-100
                    
                for i1 in range(self.symp_hospitalization.shape[0]):
                
                    if((age >= self.symp_hospitalization[i1, 0])&(age <= self.symp_hospitalization[i1, 1])):
                        
                        self.rate_array[3] = (1 - self.symp_hospitalization[i1,2] * (1 - self.prop_asymp))/(self.incub_days - self.l_days)
            
                        self.rate_array[5] = (self.symp_hospitalization[i1,2]*(1 - self.prop_asymp))/(self.incub_days - self.l_days)

                        self.rate_array[10] = (self.a_b * (1 - self.symp_hospitalization[i1,2]) + self.symp_hospitalization[i1,2]) * (1 - self.prop_asymp)/(self.incub_days - self.l_days)

                        self.rate_array[11] = (1 - (self.a_b * (1 - self.symp_hospitalization[i1,2]) + self.symp_hospitalization[i1,2]) *  (1 - self.prop_asymp))/(self.incub_days - self.l_days)
                        
                        self.rate_array[12] = (self.hosp_scale * self.symp_hospitalization[i1,2])/self.qih_days
                        
                        self.rate_array[13]= (1 - self.hosp_scale * self.symp_hospitalization[i1,2])/self.qir_days
           
                
                for i2 in range(self.percent_dead_recover_days.shape[0]):
                    if((age >= self.percent_dead_recover_days[i2,0])&(age <= self.percent_dead_recover_days[i2,1])):
                        self.rate_array[14] = (1 - (self.dead_scale * self.percent_dead_recover_days[i2,risk + 2]/100))/(self.percent_dead_recover_days[i2, 5])

                        self.rate_array[15] = (self.dead_scale * self.percent_dead_recover_days[i2,risk + 2]/100)/(self.percent_dead_recover_days[i2, 4])


                # Initialize a new Q-matrix that will change over the simulation
                Q_new = np.zeros((self.num_state, self.num_state))    

                for i3 in range(len(self.rates_indices)): 
                    Q_new[self.rates_indices[i3]] = self.rate_array[i3]            

                row_sum = np.sum(Q_new, 1)

                for i4 in range(len(row_sum)):
                    Q_new[self.diag_indices[i4]] = row_sum[i4]*(-1)     

                pop_dis_b = self.pop_dist_sim[self.t - 1][risk][age].reshape((1, self.num_state))
               
                self.pop_dist_sim[self.t][risk][age] = pop_dis_b + np.dot(pop_dis_b, (Q_new * self.dt))
                   
                # self.num_diag[self.t][risk][age] = (pop_dis_b[0,3] * self.dt * self.rate_array[7])+ (pop_dis_b[0,2] * self.dt *  self.rate_array[5])

                self.num_hosp[self.t][risk][age] = (pop_dis_b[0,6] * self.dt *  self.rate_array[12])
     
                self.num_dead[self.t][risk][age] = (pop_dis_b[0,7] * self.dt *  self.rate_array[15])

                self.num_base_test[self.t][risk][age] = pop_dis_b[0,3] * self.dt * self.rate_array[7]

                self.num_uni_test[self.t][risk][age] = (pop_dis_b[0,1] + pop_dis_b[0,2]) * self.dt * a_u
                
                self.num_trac_test[self.t][risk][age] = (pop_dis_b[0,1] + (pop_dis_b[0,1] + pop_dis_b[0,2]) * self.dt * a_u) * self.dt * (1 - a_u) * a_c
                
                self.num_hosp_test[self.t][risk][age] = pop_dis_b[0,2] * self.dt * self.rate_array[5]

            self.num_diag[self.t] = self.num_base_test[self.t] + self.num_uni_test[self.t] + self.num_hosp_test[self.t]

        # update total number of diagnosis, hospitalizations and deaths
            self.tot_num_diag[self.t] = self.tot_num_diag[self.t - 1] + np.sum(self.num_diag[self.t])
            self.tot_num_hosp[self.t] = self.tot_num_hosp[self.t - 1] + np.sum(self.num_hosp[self.t])
            self.tot_num_dead[self.t] = self.tot_num_dead[self.t - 1] +np.sum(self.num_dead[self.t])
            
    # Function to run the simulation until total number of diagnosis match with the first observed data
    def dryrun(self):

        # Initialzing simulation population distribution by age and risk
        for risk in range(self.tot_risk):
            for age in range (self.tot_age):
                self.pop_dist_sim[self.t, risk, age, 0] = self.init_pop_dist[age, risk + 1]

        # Randomly assign risk and age to latent compartment ###### wont't it change the results???
        risk = 1
        age = 50

        # Start with only one person in latent period until the total number of diagnosis match with first reported case
        self.pop_dist_sim[self.t, risk, age, 1] = 1 # L compartment
        for i in range(2, self.num_state):
            self.pop_dist_sim[self.t, risk, age, i] = 0  # E I Q_L Q_E Q_I H R D compartments
        self.pop_dist_sim[self.t, risk, age, 0] = self.pop_dist_sim[self.t, risk, age, 0] - np.sum(self.pop_dist_sim[self.t, risk, age, 1: self.num_state]) 
       
        while(self.tot_num_diag[self.t] < self.dry_run_end_diag):
            self.t += 1
            self.simulation_base(action_t = [0,0,0], beta = self.beta_max)
            
      
    # Function to run the simulation until the last day of observed data
    def sim_bf_rl_dry_run(self):
        print('sim before rl dry run begins')
        while self.t != (self.days_of_simul_pre_sd * self.inv_dt):
            self.t += 1
            self.simulation_base(action_t = [0, 0, 0], beta = self.beta_max)
            self.output_result()   
  
        while self.t !=  ((self.days_of_simul_post_sd + self.days_of_simul_pre_sd )* self.inv_dt):
            self.t += 1
            self.simulation_base(action_t = [0, 0, 0], beta = self.beta_min)
            self.output_result()    
               
    def reset_sim(self):
        print("reset_sim begin")
        self.d = 0
        self.t = 0  
        self.rate_array = np.empty([16 ,1])     # initialize rate array
        
        # Initialize measures for epidemics
        self.num_diag = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))                       # number of diagnosis
        self.num_dead = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))                       # number of deaths
        self.num_hosp = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))                       # number of hospitalizations
        
        self.pop_dist_sim = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age, self.num_state))  # population distribution by risk, age and epidemic state
        
        self.num_base_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))  
        self.num_uni_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))  
        self.num_trac_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))  
        self.num_hosp_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))  
        
        self.tot_num_diag = np.zeros(self.T_total + 1)      # total number of diagnosis
        self.tot_num_dead = np.zeros(self.T_total + 1)      # total number of deaths
        self.tot_num_hosp = np.zeros(self.T_total + 1)      # total number of hospitalizations
        
        # after dry run, total number of diagnosis should match with data 
        self.dryrun()   
        
        self.pop_dist_sim[0] = self.pop_dist_sim[self.t]
        self.num_diag[0] = self.num_diag[self.t]
        self.num_hosp[0] = self.num_hosp[self.t]
        self.num_dead[0] = self.num_dead[self.t]
        self.tot_num_diag[0] = self.tot_num_diag[self.t]
        self.tot_num_dead[0] = self.tot_num_dead[self.t]
        self.tot_num_hosp[0] = self.tot_num_hosp[self.t]
        self.t = 0
       
        self.output_result()      # record day 0                                           
        print("reset_sim end")

    def reset_rl(self):
        print("reset rl begin")
        # Initialize immediate reward
        self.imm_reward = np.zeros(self.T_total + 1)
        self.Final_VSL = np.zeros(self.T_total + 1) 
        self.Final_SAL = np.zeros(self.T_total + 1)
        self.Final_TST = np.zeros(self.T_total + 1)
        self.cost_test_u = np.zeros(self.T_total + 1)
        self.cost_test_c = np.zeros(self.T_total + 1)
        self.cost_test_b = np.zeros(self.T_total + 1)
        self.rate_unemploy = np.zeros(self.T_total + 1)
        # self.policy = np.zeros((self.T_total + 1, 3))
        self.sim_bf_rl_dry_run()                        # rl dry run until current data
        self.decison_making_day = self.t
        self.rate_unemploy[self.t] = gv.init_unemploy        # assign initial unemployment rate
        print("reset rl end")

        
def setup_COVID_sim(path):
    state = "NY"                # insert state
    inv_dt = 10                 # insert time steps within each day
    T_max_ = 365                # insert maximum simulation time period since the dry run was done
    lead_time_ = 0              # insert the time period before the action takes effect 
    time_unit_ = 'day'          # you want to model the simualtion for a couple of days, months, years 
    beta_user_defined_ = 0      # insert transmission parameter to simulate; default: 0 (in case user wanna demonstrate some tranmission parameters)

    gv.setup_global_variables(state, inv_dt,  T_max_, lead_time_, time_unit_, beta_user_defined_, path)

def run_COVID_sim(decision, path):

    sample_model = CovidModel(path)
    while sample_model.t < sample_model.T_total:
        print("##### step begin #####")
        sample_model.t += 1 
        print('t', sample_model.t)
        day = int(sample_model.rl_counter/sample_model.inv_dt)
        print('day', day)
        print('d', sample_model.d)
        print('rl_counter', sample_model.rl_counter)
        sample_model.step(action_t = decision[day-1], beta = gv.beta_before_sd)
        print("##### step end ##### \n")
 
    sample_model.op_ob.plot_decision_output_1(t = sample_model.decison_making_day, inv = sample_model.inv_dt )
    sample_model.op_ob.plot_decision_output_2(t = sample_model.decison_making_day, inv = sample_model.inv_dt )
    sample_model.op_ob.plot_decision_output_3(t = sample_model.decison_making_day, inv = sample_model.inv_dt )
    sample_model.op_ob.plot_cum_output()
    sample_model.op_ob.plot_time_output()

if  __name__ == "__main__":
    path = os.getcwd()
    setup_COVID_sim(path)
    decision = gv.decision
    run_COVID_sim(decision, path)
    
    
    

    