import numpy as np 
import pandas as pd
import os
import datetime

import global_var as gv
import outputs as op
import ImageMerge as merge


class CovidModel():
    def __init__(self, path, decision):
        super(CovidModel, self).__init__()

        self.beta_max = gv.beta_before_sd  # max transmission rate (normal pre-COVID 19)
        self.beta_min = gv.beta_after_sd   # min transmission rate ()

        """if gv.beta_user_defined != 0:      # user-defined transmission rate
            self.beta = gv.beta_user_defined
        else: 
            self.beta = self.beta_max"""
        
        self.enter_state = gv.enter_state  # two letter abbreviation of the state you want to model

        self.tot_risk = gv.tot_risk        # total risk group: female, male
        self.tot_age = gv.tot_age          # total age group: 101, by age
        
        self.T = gv.T_max          # decision making time period
        self.inv_dt = gv.inv_dt    # n time steps in one day
        self.dt = 1/self.inv_dt    # inverse of n time step
        
        """ ##### this one would be incorporated in the later stages of modeling, not used now
        self.lead_time = gv.lead_time          # the time period before the action takes effect"""  

        ### simulation related variables; won't change during the simulation 
        self.Q = gv.Q                                                      # a base Q-matrix 
        self.num_state = self.Q.shape[0]                                   # number of states during the simulation
        self.rates_indices = gv.rates_indices                              # rate matrix for calculating transition rate
        self.diag_indices = gv.diag_indices                                # diagonal matrix for calculating transition rate
        self.symp_hospitalization = gv.symp_hospitalization_v              
        self.percent_dead_recover_days = gv.percent_dead_recover_days_v  
        self.init_pop_dist = gv.pop_dist_v                                 # initial population distribution 
        self.tot_pop = np.sum(self.init_pop_dist)                          # total number of population by State
        self.dry_run_end_diag = gv.dry_run_end_diag                        # after dry run, total number of diagnosis should match with data
        self.days_of_simul_pre_sd = gv.days_of_simul_pre_sd                # number of days before social distancing
        self.days_of_simul_post_sd = gv.days_of_simul_post_sd              # number of days after social distancing before the end of observed data

        self.input_list_const = gv.input_list_const_v                      # input parameters for reading the below parameters
        self.l_days =  self.input_list_const.loc['Days_L', 'value']        # latent period duration
        self.prop_asymp = self.input_list_const.loc['Prop_Asymp', 'value'] # proportion of cases that never show symptoms
        self.incub_days = self.input_list_const.loc['Days_Incub', 'value'] # incubation period duration 
        self.a_b = self.input_list_const.loc['a_b', 'value']               # symptom based testing rate
        self.ir_days = self.input_list_const.loc['Days_IR', 'value']       # time from onset of symptoms to recovery 
        self.qih_days = self.input_list_const.loc['Days_QiH', 'value']     # time from onset of symptoms to hospitalization
        self.qir_days = self.input_list_const.loc['Days_QiR', 'value']     # time from diagnosis to recovery 
        self.second_attack_rate = self.input_list_const.loc['Second_attack', 'value']/100
        self.hosp_scale = gv.hosp_scale                                    # hospitalization scale factor
        self.dead_scale = gv.dead_scale                                    # death scale factor 

        # simulation time period since dry run
        self.T_total = self.inv_dt * (self.T + self.days_of_simul_pre_sd + self.days_of_simul_post_sd) # simulation time period from dry run
      
        # rl related parameters; won't change during the simulation 
        self.lab_for = gv.lab_for                                 # labor force participation rate 
        self.VSL = gv.VSL                                         # value of statistical life by age (1-101)
        self.md_salary = gv.md_salary                             # median salary per time step 
        self.K_val = gv.K_val                                     # coefficient for calculating unemployment rate
        self.A_val = gv.A_val                                     # coefficient for calculating unemployment rate
        self.duration_unemployment = gv.duration_unemployment     # duration from social distaning to reaching maximum of unemployment rate

        self.cost_tst = gv.test                                   # cost of testing per person ([0]: symptom-based, 
                                                                  # [1]: contact tracing, [2]: universal testing)

        # initialize observation 
        self.op_ob = op.output_var(int(self.T_total/self.inv_dt) + 1, state = self.enter_state, cwd = path, policy = decision)

        self.reset_rl()                                           # initialize rl 
        self.reset_sim()                                          # reset the simulation 
    
    # Function to simulate compartment transition, calculate immediate reward function and output result
    # Input parameter:
    # beta = a float value representing transmission rate for the simulation 
    def step(self, action_t):
        self.set_action(action_t)
        self.simulation_base() 
        self.calc_imm_reward()
        self.output_result() 
                                 
    # Function to output result for plotting
    # Input parameters: 
    # None
    def output_result(self):
        if self.t % self.inv_dt == 0: 
            gv.prog_bar.next()               
            self.op_ob.time_step[self.d] = self.d     # timestep (day)
            #### if plot for the day 
            indx_l = self.t - self.inv_dt + 1 # = self.t
            indx_u = self.t + 1  # = self.t + 1

            self.op_ob.num_inf_plot[self.d] = np.sum(self.num_diag[indx_l: indx_u])          # new infected for the day
            self.op_ob.num_hosp_plot[self.d] = np.sum(self.num_hosp[indx_l: indx_u])         # new hospitablization for the day
            self.op_ob.num_dead_plot[self.d] = np.sum(self.num_dead[indx_l: indx_u])         # new death for the day
            self.op_ob.cumulative_inf[self.d] =  self.tot_num_diag[self.t]                   # cumulative infections from start of simulation to the day
            self.op_ob.cumulative_hosp[self.d] = self.tot_num_hosp[self.t]                   # cumulative hospitalizations from start of simulation to the day
            self.op_ob.cumulative_dead[self.d] = self.tot_num_dead[self.t]                   # cumulative dead from start of simulation to the day
            
            self.op_ob.num_base[self.d] = np.sum(self.num_base_test[indx_l: indx_u])         # number of symptom based testing for the day
            self.op_ob.num_uni[self.d] = np.sum(self.num_uni_test[indx_l: indx_u])           # number of universal testing for the day
            self.op_ob.num_trac[self.d] = np.sum(self.num_trac_test[indx_l: indx_u])         # number of contact tracing based testing for the day
            self.op_ob.num_hop_tst[self.d] =  np.sum(self.num_hosp_test[indx_l: indx_u])     # number of hospitalized based testing for the day
            
            self.op_ob.VSL_plot[self.d] =  (np.sum(self.Final_VSL[indx_l: indx_u]))          # VSL at timestep t
            self.op_ob.SAL_plot[self.d] =  (np.sum(self.Final_SAL[indx_l: indx_u]))          # SAL at timestep t        
            self.op_ob.unemployment[self.d] = self.rate_unemploy[self.t] * 100               # unemployment rate at time step t
            self.op_ob.univ_test_cost[self.d] =  (np.sum(self.cost_test_u[indx_l: indx_u]))  # cost of universal testing for the day 
            self.op_ob.trac_test_cost[self.d] =  (np.sum(self.cost_test_c[indx_l: indx_u]))  # cost of contact tracing for the day 
            self.op_ob.bse_test_cost[self.d] =  (np.sum(self.cost_test_b[indx_l: indx_u]))   # symptom based testing for the day
            
            self.op_ob.num_diag_inf[self.d] = self.num_diag_inf[self.t]                      # Q_L + Q_E + Q_I
            self.op_ob.num_undiag_inf[self.d] = self.num_undiag_inf[self.t]                  # L + E + I
            self.d += 1
            
    # Function to convert action 
    def set_action(self, action_t):
        self.a_sd = action_t[0]
        self.T_c = action_t[1]
        self.T_u = action_t[2]   
        self.a_u = self.T_u / np.sum(self.pop_dist_sim[(self.t - 1),:,:,0:4])
        self.a_c = min(1, (self.T_c * self.second_attack_rate)/((1 - self.a_u) * np.sum(self.pop_dist_sim[(self.t - 1),:,:,1:4])))
    

    # Function to calculate immediate reward /cost
    # Input parameter:
    # action_t = an np array of size [1x3] with the values output by the RL model (a_sd, T_c, T_u)
    def calc_imm_reward(self):
        million = 1000000 # one million dollars
        
        self.calc_unemployment()
    
        tot_alive = self.tot_pop - self.tot_num_dead[self.t - 1]   # total number of alive people at time step (t - 1)
    
        # number of unemployed = total alive people at time step (t - 1) x labor force participation rate /100  
        #                        x unemployment rate / 100
        num_unemploy = tot_alive * self.rate_unemploy[self.t - 1] * self.lab_for  # rate converted to percentage

        # calculate total wage loss due to contact reducation  = number of unemployed x median wage / 1 million
        self.Final_SAL[self.t] = num_unemploy * self.md_salary * self.dt / million  
      
        # calculate total 'value of statistical life' loss due to deaths = number of newly dead x VSL (by age)
        num_dead = np.sum(self.num_dead[self.t - 1], axis = 0)
        self.Final_VSL[self.t]  = np.sum(np.dot(num_dead , self.VSL)) 
       
        # calculate cost of testing 
        self.cost_test_b[self.t] =  self.cost_tst[0] * np.sum(self.num_base_test[self.t]) /million
        self.cost_test_c[self.t] =  self.dt * self.cost_tst[1] * self.a_c * (1 - self.a_u) * np.sum(self.pop_dist_sim[(self.t - 1),:,:,1:4]) /(self.second_attack_rate * million)
        self.cost_test_u[self.t] =  self.dt * self.cost_tst[2] * self.T_u / million
        self.Final_TST[self.t] = self.cost_test_u[self.t] + self.cost_test_c[self.t] + self.cost_test_b[self.t] 

        # calculate immeidate reward 
        self.imm_reward[self.t] = -1 * (self.Final_VSL[self.t]  + self.Final_SAL[self.t] + self.Final_TST[self.t])
 
     

    # Function to calculate unemployment change
    # Input parameter:
    # a_sd = a float value represents proportion of contact reduction
    def calc_unemployment(self):
        
        y_p = self.rate_unemploy[self.t-1]
        
        K = max(self.a_sd * self.K_val, y_p)

        A = max(self.A_val, min(self.a_sd * self.K_val, y_p))

        u_plus = (K - A)/self.duration_unemployment

        u_minus = 0.5 * (K - A)/self.duration_unemployment
        if y_p == K:
            self.rate_unemploy[self.t] = y_p - u_minus * self.dt
        else:
            self.rate_unemploy[self.t] = y_p + u_plus *  self.dt

       

    # Function to calculate transition rates (only for the rates that won't change by risk or age)
    # Input parameter:
    # action_t = an np array of size [1x3] with the values output by the RL model (a_sd, T_c, T_u)
    def set_rate_array(self):
        # rate of S -> L
        beta_sd = self.beta_min + (1 - self.a_sd) * (self.beta_max - self.beta_min)
        self.rate_array[0] = (beta_sd * np.sum(self.pop_dist_sim[(self.t - 1),\
                              :,:,2:4]))/(np.sum(self.pop_dist_sim[(self.t - 1), :,:,0:9]))
        # rate of L -> E
        self.rate_array[1] = 1/self.l_days
        # rate of L -> Q_L
        self.rate_array[2] = self.a_u + ((1 - self.a_u)*self.a_c)
        # rate of E -> Q_E
        self.rate_array[4] = self.a_u + ((1 - self.a_u)*self.a_c)
        # rate of E -> Q_I
        self.rate_array[6] = self.prop_asymp/(self.incub_days - self.l_days)
        # rate of I -> Q_I
        self.rate_array[7] = self.a_b
        # rate of I -> R
        self.rate_array[8] = ((self.a_u + (1-self.a_u)*self.a_c)) + 1/self.ir_days  
        # rate of Q_L -> Q_E
        self.rate_array[9] = 1/self.l_days


    # Function to perform the simulation
    # Input parameters for this function
    # action_t = an np array of size [1x3] with the values output by the RL model (a_sd, T_c, T_u)
    
    # 0	1	2	3	4	5	6	7	8	9 compartments
    # S	L	E	I	Q_L	Q_E	Q_I	H	R	D compartments
    def simulation_base(self):
        # Calculate transition rate that won't change during the for loop
        self.set_rate_array()

        for risk in range(self.tot_risk): # for each risk group i.e, male(0) and female(1)

            for age in range (self.tot_age): # for each age group i.e., from 0-100
                    
                for i1 in range(self.symp_hospitalization.shape[0]): 
                
                    if((age >= self.symp_hospitalization[i1, 0])&(age <= self.symp_hospitalization[i1, 1])):
                        
                        # rate of E -> I 
                        self.rate_array[3] = (1 - self.symp_hospitalization[i1,2] * (1 - self.prop_asymp))/(self.incub_days - self.l_days)
                        # rate of E -> Q_I
                        self.rate_array[5] = (self.symp_hospitalization[i1,2]*(1 - self.prop_asymp))/(self.incub_days - self.l_days)
                        # rate of Q_E -> Q_I
                        self.rate_array[10] = (self.a_b * (1 - self.symp_hospitalization[i1,2]) + self.symp_hospitalization[i1,2]) * (1 - self.prop_asymp)/(self.incub_days - self.l_days)
                        # rate of Q_E -> R
                        self.rate_array[11] = (1 - (self.a_b * (1 - self.symp_hospitalization[i1,2]) + self.symp_hospitalization[i1,2]) *  (1 - self.prop_asymp))/(self.incub_days - self.l_days)
                        # rate of Q_I to H
                        self.rate_array[12] = (self.hosp_scale * self.symp_hospitalization[i1,2])/self.qih_days
                        # rate of Q_I to R
                        self.rate_array[13]= (1 - self.hosp_scale * self.symp_hospitalization[i1,2])/self.qir_days
           
                
                for i2 in range(self.percent_dead_recover_days.shape[0]):
                    if((age >= self.percent_dead_recover_days[i2,0])&(age <= self.percent_dead_recover_days[i2,1])):
                        # rate of H to D
                        self.rate_array[14] = (1 - (self.dead_scale * self.percent_dead_recover_days[i2,risk + 2]/100))/(self.percent_dead_recover_days[i2, 5])
                        # rate of H to R
                        self.rate_array[15] = (self.dead_scale * self.percent_dead_recover_days[i2,risk + 2]/100)/(self.percent_dead_recover_days[i2, 4])


                # Initialize a new Q-matrix that will change over the simulation
                Q_new = np.zeros((self.num_state, self.num_state))    

                for i3 in range(len(self.rates_indices)): 
                    Q_new[self.rates_indices[i3]] = self.rate_array[i3]            

                row_sum = np.sum(Q_new, 1)

                for i4 in range(len(row_sum)):
                    Q_new[self.diag_indices[i4]] = row_sum[i4]*(-1)     
                
                pop_dis_b = self.pop_dist_sim[self.t - 1][risk][age].reshape((1, self.num_state))
                # population distribution state transition 
                self.pop_dist_sim[self.t][risk][age] = pop_dis_b + np.dot(pop_dis_b, (Q_new * self.dt))
                # number of new hospitalized at time step t
                self.num_hosp[self.t][risk][age] = pop_dis_b[0,6] * self.dt *  self.rate_array[12]
                # number of new death at time step t
                self.num_dead[self.t][risk][age] = pop_dis_b[0,7] * self.dt *  self.rate_array[15]
                # number of diagnosis through symptom based testing
                self.num_base_test[self.t][risk][age] = pop_dis_b[0,3] * self.dt * self.rate_array[7] + pop_dis_b[0,2] * self.dt * self.rate_array[5]
                # number of diagnosis through universal testing
                self.num_uni_test[self.t][risk][age] = (pop_dis_b[0,1] + pop_dis_b[0,2] + pop_dis_b[0,3]) * self.dt * self.a_u
                # number of diagnosis through contact tracing
                self.num_trac_test[self.t][risk][age] = (pop_dis_b[0,1] + pop_dis_b[0,2] + pop_dis_b[0,3]) * self.dt * (1 - self.a_u) * self.a_c
                
        # the total number of diagnosis
        self.num_diag[self.t] = self.num_base_test[self.t] + self.num_trac_test[self.t] + self.num_uni_test[self.t]
            # update total number of diagnosis, hospitalizations and deaths
        self.tot_num_diag[self.t] = self.tot_num_diag[self.t - 1] + np.sum(self.num_diag[self.t])
        self.tot_num_hosp[self.t] = self.tot_num_hosp[self.t - 1] + np.sum(self.num_hosp[self.t])
        self.tot_num_dead[self.t] = self.tot_num_dead[self.t - 1] +np.sum(self.num_dead[self.t])
            
        self.num_diag_inf[self.t] = np.sum(self.pop_dist_sim[(self.t - 1),:,:,4:7])
        self.num_undiag_inf[self.t] = np.sum(self.pop_dist_sim[(self.t - 1),:,:,1:4])

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
            self.simulation_base()

    # Function to run the simulation until the last day of observed data
    # Input parameter:
    # None
    def sim_bf_rl_dry_run(self):
        # print('sim before rl dry run begins')
        t = 1
        # simulate before social distancing measures
        while t <= (self.days_of_simul_pre_sd * self.inv_dt):
            self.t += 1
            self.step(action_t = [0, 0, 0])
            t += 1

        t = 1
        # simulate after social distancing measure
        while t <=  (self.days_of_simul_post_sd * self.inv_dt):
            self.t += 1
            self.step(action_t = [1, 0, 0])
            t += 1

    # Function to intialize simulation, do dry run and any simulation before the decision making   
    # Input parameter:
    # None
    def reset_sim(self):
        # print("reset_sim begin")
        self.d = 0
        self.t = 0  
        self.rate_array = np.zeros([16 ,1])     # initialize rate array
        
        # Initialize measures for epidemics
        self.num_diag = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))        # number of diagnosis
        self.num_dead = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))        # number of deaths
        self.num_hosp = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))        # number of hospitalizations
        
        self.pop_dist_sim = np.zeros((self.T_total + 1, self.tot_risk, \
                                      self.tot_age, self.num_state))                     # population distribution by risk, age and epidemic state

        self.num_base_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))   # number of diagnosed through symptom-based testing 
        self.num_uni_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))    # number of diagnosed through universal testing
        self.num_trac_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))   # number of diagnosed through contact tracing
        self.num_hosp_test = np.zeros((self.T_total + 1, self.tot_risk, self.tot_age))   # number of diagnosed through hospitalization

        self.tot_num_diag = np.zeros(self.T_total + 1)                                   # cumulative diagnosed
        self.tot_num_dead = np.zeros(self.T_total + 1)                                   # cumulative deaths
        self.tot_num_hosp = np.zeros(self.T_total + 1)                                   # cumulative hospitalizations
        
        self.num_diag_inf = np.zeros(self.T_total + 1)                                   # Q_L + Q_E + Q_I
        self.num_undiag_inf = np.zeros(self.T_total + 1)                                 # L + E + I

        # initialize action
        self.a_sd = 0
        self.a_c = 0
        self.a_u = 0
        self.T_c = 0
        self.T_u = 0

        # after dry run, total number of diagnosis should match with data 
        self.dryrun()   
        
        # re-initialize all the below parameters
        self.pop_dist_sim[0] = self.pop_dist_sim[self.t]
        self.num_diag[0] = self.num_diag[self.t]
        self.num_hosp[0] = self.num_hosp[self.t]
        self.num_dead[0] = self.num_dead[self.t]
        self.tot_num_diag[0] = self.tot_num_diag[self.t]
        self.tot_num_dead[0] = self.tot_num_dead[self.t]
        self.tot_num_hosp[0] = self.tot_num_hosp[self.t]
        self.num_base_test[0] = self.num_base_test[self.t]
        self.num_uni_test[0] = self.num_uni_test[self.t]
        self.num_trac_test[0] = self.num_trac_test[self.t]
        self.num_hosp_test[0] = self.num_hosp_test[self.t]
        self.num_diag_inf[0] = self.num_diag_inf[self.t]
        self.num_undiag_inf[0] = self.num_undiag_inf[self.t]
        self.rate_unemploy[0] = gv.init_unemploy        # assign initial unemployment rate     

        # reset time
        self.t = 0

        self.output_result()      # record day 0   

        self.sim_bf_rl_dry_run()                             # rl dry run until current observed data
        self.decison_making_day = self.t                     # the start day of decision making                        
        # print("reset_sim end")

    # initialize decision making 
    def reset_rl(self):
        # print("reset rl begin")
        # Initialize immediate reward related parameters
        self.imm_reward = np.zeros(self.T_total + 1)
        self.Final_VSL = np.zeros(self.T_total + 1) 
        self.Final_SAL = np.zeros(self.T_total + 1)
        self.Final_TST = np.zeros(self.T_total + 1)
        self.cost_test_u = np.zeros(self.T_total + 1)
        self.cost_test_c = np.zeros(self.T_total + 1)
        self.cost_test_b = np.zeros(self.T_total + 1)
        self.rate_unemploy = np.zeros(self.T_total + 1)
        self.policy = np.zeros((self.T_total + 1, 3))  # not used now 
   
        # print("reset rl end")

        
def setup_COVID_sim(state, path):             
    inv_dt = 10                 # insert time steps within each day
    T_max_ = 365                # insert maximum simulation time period since the dry run was done
    lead_time_ = 0              # insert the time period before the action takes effect 
    time_unit_ = 'day'          # you want to model the simualtion for a couple of days, months, years 
    beta_user_defined_ = 0      # insert transmission parameter to simulate; default: 0 
                                #(in case user wanna demonstrate some tranmission parameters)

    gv.setup_global_variables(state, inv_dt,  T_max_, lead_time_, time_unit_, beta_user_defined_, path)

def run_COVID_sim(decision, path, verbose = 'Y', write = 'N'):

    sample_model = CovidModel(path, decision)
    i = 0
    d_m = decision[i]
    while sample_model.t < sample_model.T_total:
        # print("##### step begin #####")
        # print('The code is running')
        sample_model.t += 1 
        # print('t', sample_model.t)
        if i % sample_model.inv_dt == 0:
            d_m = decision[i//sample_model.inv_dt]
        sample_model.step(action_t = d_m)
        i += 1
        # print("##### step end ##### \n")

    gv.prog_bar.finish()

    df1 = sample_model.op_ob.plot_decision_output_1()
    df2= sample_model.op_ob.plot_decision_output_2(gv.acutal_unemp)
    df3 = sample_model.op_ob.plot_decision_output_3()
    df4 = sample_model.op_ob.plot_cum_output(gv.actual_data)
    df5 = sample_model.op_ob.plot_decison()
    # # sample_model.op_ob.plot_time_output()
 
    if write == 'Y' or write == 'y':
        sample_model.op_ob.write_output(df1, df2, df3, df4, df5)
    else:
        pass
    
    merge.merge_image()
    print('Finished the run')

if  __name__ == "__main__":
    path = os.getcwd()
    state = 'NY' # default is New York
    print('This is a model for State of New York')
    # state = input('insert two letter abbreviation for the State that you want to model (e.g.: NY for New York): ')  # insert two letter abbreviation state that you want to model
    setup_COVID_sim(state, path) 
    print('Do you want to test a decision (Y or N)?')
    print('If you choose N (No), it will assume social distancing measures' \
          'as of May 3rd are maintained for the next 52 weeks,'\
          'and testing is only through baseline symptom-based testing, '\
          'i.e., no contact tracing and testing, and no universal testing')
    bol_ = 'N'
    bol_ = input('Enter Y or N: ')
    print('\n')
    if bol_ == 'Y' or bol_ =='y':
        print("Enter decision choice for social distancing as 'percent reduction in contacts compared to a normal pre-COVID situation' "\
              "for weeks 1 through 52 as Start week1," \
              'end week1, decision1, Start week2, end week2, decision2,……,and so on, for as many options as you need')

        print('Example 1: if you want week 1 to have 50'+'%'+' reduction, '\
              'and weeks 2 to 52 to have 30' +'%'+' reduction, enter 1,1,0.5,2,52,0.3')

        print('Example 2: if you want weeks 1 to 5 to have 50'+'%'+' reduction, '\
              'weeks 6 to 10 to have 30'+'%'+' reduction, and weeks 11 to 52 as 0'+'%'+' reduction,, '\
              'enter 1,5,0.5,6,10,0.3,11,52,0')
        #print('NOTE: The maximum contact reduction is ' + str(int(100)) +'%')
        a_sd_str = input('Enter value here: ')
        print('\n')
        print("Enter decision choice for 'contact tracing and testing capcity per day'"\
              "for weeks 1 through 52 as Start week1, end week1, decision1, Start week2, end week2, decision2,.…, and so on, for as many options as you need")

        print('Example 1: If you can do 100 tests per day for weeks 1 to 10, and 1000 tests per day for weeks 11 to 52, '\
              'enter 1,10,100,11,52,1000')

        print('Example 2: If you can do 100 tests per day for week 1, 200 tests per day for week 2, '\
              'and 1000 tests per day for weeks 3 to 52, enter 1,1,100,2,2,200,3,52,1000')

        a_c_str = input('Enter value here: ')
        print('\n')
        print("Enter decision choice for 'testing capacity per day for universal testing of population' "\
              'for weeks 1 through 52 as Start week1, end week1, decision1, Start week2, end week2, decision2, ……, and so on, for as many options as you need')

        print('Example 1: If you can do 100 tests per day for weeks 1 to 10, and 1000 tests per day for weeks 11 to 52, '\
              'enter 1,10,100,11,52,1000')

        print('Example 2: If you can do 100 tests per day for week 1, 200 tests per day for week 2, '\
              'and 1000 tests per day for weeks 3 to 52, enter 1,1,100,2,2,200,3,52,1000')
        a_u_str = input('Enter value here: ')
        print('\n')
        gv.decision = gv.get_decisions(a_sd_str,a_c_str,a_u_str,gv.T_max)
     
    else:
        gv.decision = gv.format_year(gv.decision_week, gv.T_max)
   
    #print("NOTE: plots are automatically saved in the folder\n")
    print('Do you want to write results into excel file?')
    #print('NOTE: writing it takes a longer time to process')
    write_ = 'N'
    write_ = input('Enter Y or N (e.g.: Y ): ')
    
    run_COVID_sim(path = path, decision = gv.decision, write = write_)

    
    
    

    
