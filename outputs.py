import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import pdb
import global_var as gv
import pandas as pd
from matplotlib.ticker import FuncFormatter

class output_var:

    def __init__(self, sizeofrun, state, cwd, policy):
        self.time_step = np.zeros(sizeofrun)
        self.action_plot = np.zeros(sizeofrun)
        self.a_sd_plot = np.zeros(sizeofrun)
        self.num_inf_plot = np.zeros(sizeofrun)  #reported cases             
        self.num_hosp_plot = np.zeros(sizeofrun)  #severe cases
        self.num_dead_plot = np.zeros(sizeofrun)
        self.VSL_plot = np.zeros(sizeofrun)
        self.SAL_plot = np.zeros(sizeofrun)
        self.cumulative_inf = np.zeros(sizeofrun)
        self.cumulative_hosp = np.zeros(sizeofrun)
        self.cumulative_dead = np.zeros(sizeofrun)
        self.unemployment = np.zeros(sizeofrun)
        self.univ_test_cost = np.zeros(sizeofrun)
        self.trac_test_cost = np.zeros(sizeofrun)
        self.bse_test_cost = np.zeros(sizeofrun)
        self.num_base = np.zeros(sizeofrun)
        self.num_uni = np.zeros(sizeofrun)
        self.num_trac = np.zeros(sizeofrun)
        self.num_hop_tst =  np.zeros(sizeofrun)
        self.start_d, self.sd_d, self.decision_d  = gv.read_date(state, cwd)
        self.policy_plot = policy
        self.num_diag_inf = np.zeros(sizeofrun)
        self.num_undiag_inf = np.zeros(sizeofrun)

        self.dpi = 150
        

    def write_output(self, df1, df2, df3, df4, df5, choice = 1):
        if choice == 1:
            writer = pd.ExcelWriter('final_result.xlsx', engine = 'xlsxwriter')

            df1.to_excel(writer, sheet_name = 'VSL')
            df2.to_excel(writer, sheet_name = 'Unemployment')
            df3.to_excel(writer, sheet_name = 'Testing')
            df4.to_excel(writer, sheet_name = 'Summary')
            df5.to_excel(writer, sheet_name = 'Decision choice')
            
            writer.save()
        else:
            df_data = np.array( [self.num_inf_plot, self.num_hosp_plot, self.num_dead_plot, self.VSL_plot, self.SAL_plot, \
                    self.cumulative_inf, self.cumulative_hosp, self.cumulative_dead, self.unemployment, \
                    self.univ_test_cost, self.trac_test_cost, self.bse_test_cost, \
                    self.num_base, self.num_uni, self.num_trac, self.num_hop_tst])
            df_name = ['number of diagnosed', 'number of hospitalization', 'number of deaths', 'VSL', 'wage loss',\
                    'cumumlative diagnosed', 'cumulative hospitalization', 'cumulative deaths', \
                    'unemployment rate', 'cost of universal testing', 'cost of contact tracing', 'cost of base testing',\
                    'number of diagnosed by base tesing', 'number of diagnosed by universal testing', \
                    'number of diagnosed by contact tracing', 'number of diagnosed by hospitalization']
            df = pd.DataFrame(data = df_data.T, index = pd.date_range(start= self.start_d, periods= df_data.shape[1]),columns = df_name)

            df.to_csv("final_result.csv")
      
    def plot_decision_output_1(self):
        plt.style.use('seaborn')
        df_data = np.array([self.VSL_plot, self.num_dead_plot])
        df_name = ['Value of statistical life-year (VSL) loss', 'Number of deaths']
        df = pd.DataFrame(data = df_data.T, index = pd.date_range(start= self.start_d, periods= df_data.shape[1]),columns = df_name)
        
        ### plot figure: number of deaths VS vsl loss 
        fig, ax = plt.subplots(2, 1)
        df.plot(y = 'Number of deaths', title = 'Number of new deaths per day', use_index = True, ax = ax[0], \
                legend = False, fontsize = 10, color ='r')
        df.plot(y = 'Value of statistical life-year (VSL) loss', title = 'Value of statistical life-year (VSL) loss per day', \
                use_index = True, ax = ax[1], legend = False, fontsize = 10)
        ax[1].set_ylabel("US dollars (in millions)")
        plt.subplots_adjust(hspace = 0.5)

        # create an arrow
        bbox = dict(boxstyle="round", fc="0.8")
        arrowprops = dict(arrowstyle = "->",connectionstyle = "angle,angleA=0,angleB=90,rad=5")
        offset = 72
        
        y2 = df.loc[self.decision_d]['Value of statistical life-year (VSL) loss']
        ax[1].annotate('Start of decision making',(self.decision_d,y2),xytext=(0.5*offset, 0.3*offset), \
          textcoords='offset points',bbox=bbox, arrowprops=arrowprops)

        y1 = df.loc[self.decision_d]['Number of deaths']
        ax[0].annotate('Start of decision making',(self.decision_d,y1),xytext=(0.5*offset, 0.3*offset), \
          textcoords='offset points',bbox=bbox, arrowprops=arrowprops)
        plt.savefig('1.png',dpi = self.dpi)
        plt.close()
        
        return df
       


    def plot_decision_output_2(self, actual_unemp):
        plt.style.use('seaborn')
        date = pd.date_range(start= self.start_d, periods= self.unemployment.shape[0])
        df_data = np.array([date, self.SAL_plot, self.unemployment])
        df_name = ['Date', 'Wage loss', 'Assumption under selected social distancing']
        df = pd.DataFrame(data = df_data.T, index = None, columns = df_name)
        
        fig, ax = plt.subplots(2, 1)
        day = pd.Timestamp(self.decision_d)
        # df.loc[df['Date'] >= self.start_d].plot(x = 'Date', y = 'Assumption under selected social distancing', \
        #           use_index = True, ax = ax[0], fontsize = 10, marker= '.', linestyle = '--')
        df.loc[df['Date'] >=day].plot(x = 'Date', y = 'Assumption under selected social distancing', \
                  use_index = True, ax = ax[0], fontsize = 10, marker= '.', linestyle = '--')

        actual_unemp.loc[actual_unemp['Date'] >= self.start_d].plot(x = 'Date',\
                        y = 'Actual unemployment rate', ax = ax[0], fontsize = 10,marker= '.',\
                        label = 'Actual unemployment rate')
        ax[0].set_title('Unemployment rate \n (Assumption: Assumption for unemployment rate under selected social distancing)')
        ax[0].set_xlim(left = pd.Timestamp(self.start_d) )
        ax[0].set_ylabel('Rate (100%)')
        df.loc[df['Date']>= day].plot(x = 'Date', y = 'Wage loss', title = 'Wage loss per day', \
                                      use_index = True, ax = ax[1], legend = False, fontsize = 10,\
                                      marker= '.', linestyle = '--')
        
        ax[1].set_ylabel("US dollars (in millions)")
        ax[1].set_xlim(left =pd.Timestamp(self.start_d))

        # create an arrow
        bbox = dict(boxstyle="round", fc="0.8")
        arrowprops = dict(arrowstyle = "->",connectionstyle = "angle,angleA=0,angleB=70,rad=5")
        offset = 72
        
        y1 = df.loc[df['Date'] == self.decision_d]['Assumption under selected social distancing']
        ax[0].annotate('Start of decision making',(self.decision_d,y1),xytext=(0.5*offset, -0.5*offset), \
                        textcoords='offset points',bbox=bbox, arrowprops=arrowprops)
        y2 = df.loc[df['Date'] == self.decision_d]['Wage loss']
        ax[1].annotate('Start of decision making',(self.decision_d,y2),xytext=(0.5*offset, 0.5*offset), \
                        textcoords='offset points',bbox=bbox, arrowprops=arrowprops)

        plt.subplots_adjust(hspace = 0.5)
        plt.savefig('2.png',dpi = self.dpi)
        plt.close()
        df = df.set_index('Date')
        return df
        

    def plot_decision_output_3(self):
        plt.style.use('seaborn')
        fig, ax = plt.subplots(2, 1)
        df_data = np.array([self.univ_test_cost, self.trac_test_cost, self.bse_test_cost,\
                            self.num_trac, self.num_base, self.num_uni])
        df_name = ['cost of universal testing', 'cost of contact tracing', 'cost of symptom-based testing',\
                   'by contact tracing', 'by symptom-based testing', 'by universal testing']
        df = pd.DataFrame(data = df_data.T, index = pd.date_range(start= self.start_d, \
                          periods= df_data.shape[1]), columns = df_name)
        
        df.loc[self.decision_d:].plot(y = 'cost of universal testing', use_index = True, ax = ax[0], fontsize = 10)
        df.loc[self.decision_d:].plot(y = 'cost of contact tracing', use_index = True, ax = ax[0], fontsize = 10)
        df.plot(y = 'cost of symptom-based testing', use_index = True, ax = ax[0], fontsize = 10)
        ax[0].set_ylabel("US dollars (in millions)")
        ax[0].set_title("Cost of testing by type per day")
        ax[0].set_xlim(left = pd.Timestamp(self.start_d))

        df.loc[self.decision_d:].plot(y = 'by universal testing', use_index = True, ax = ax[1], fontsize = 10)
        df.loc[self.decision_d:].plot(y = 'by contact tracing', use_index = True, ax = ax[1], fontsize = 10)
        df.plot(y = 'by symptom-based testing', use_index = True, ax = ax[1], fontsize = 10)
        ax[1].set_title("Number of new diagnosis by testing type per day")
        ax[1].set_xlim(left = pd.Timestamp(self.start_d))

        bbox = dict(boxstyle="round", fc="0.8")
        arrowprops = dict(arrowstyle = "->",connectionstyle = "angle,angleA=0,angleB=70,rad=5")
        offset = 72
        ax[0].annotate('Start of decision making',(self.decision_d,0),xytext=(0.5*offset, 0.5*offset), \
                        textcoords='offset points',bbox=bbox, arrowprops=arrowprops)
        ax[1].annotate('Start of decision making',(self.decision_d,0),xytext=(0.5*offset, 0.5*offset), \
                        textcoords='offset points',bbox=bbox, arrowprops=arrowprops)

        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        plt.savefig('3.png',dpi = self.dpi)
        plt.close()
        return df

    def plot_cum_output(self, actual_data): 
        plt.style.use('seaborn')
        date = pd.date_range(start= self.start_d, periods= self.cumulative_inf.shape[0])
        df_data = np.array([date, self.cumulative_inf, self.cumulative_hosp, self.cumulative_dead, \
                            self.num_diag_inf, self.num_undiag_inf])
        df_name = ['date', 'projected cumulative diagnosis', 'projected cumulative hospitalized',\
                   'projected cumulative deaths', 'number of infected, diagnosed', 'number of infected, undiagnosed']
        
        df = pd.DataFrame(data = df_data.T, index = None, columns = df_name)
      
        fig, ax = plt.subplots()
        df.plot(x = 'date', y = 'projected cumulative diagnosis', fontsize = 10, ax = ax)
        actual_data.plot(x = 'date', y = 'actual cumulative diagnosis', fontsize = 10, ax = ax)
        plt.title("Cumulative diagnosis over time")
        plt.savefig('4.png',dpi = self.dpi)
        plt.close()

        fig1, ax1 = plt.subplots()
        df.plot(x = 'date', y = 'projected cumulative deaths', fontsize = 10, ax = ax1)
        actual_data.plot(x = 'date', y = 'actual cumulative deaths', fontsize = 10, ax = ax1)
        plt.title("Cumulative deaths over time")
        plt.savefig('5.png',dpi = self.dpi)
        plt.close()

        fig2, ax2 = plt.subplots()
        df.plot(x = 'date', y = 'projected cumulative hospitalized', fontsize = 10, ax = ax2)
        actual_data.plot(x = 'date', y = 'actual cumulative hospitalized', fontsize = 10, ax = ax2)
        plt.title("Cumulative hospitalizations over time")
        plt.savefig('6.png',dpi = self.dpi)
        plt.close()

       
        fig3, ax3 = plt.subplots()
        df.plot(x = 'date', y = 'number of infected, diagnosed', fontsize = 10, ax = ax3)
        df.plot(x = 'date', y = 'number of infected, undiagnosed', fontsize = 10, ax = ax3)
        plt.title("Number of people with infection")
        plt.savefig('7.png',dpi = self.dpi)

        df = df.set_index('date')
        return df

    def plot_decison(self):
        plt.style.use('seaborn')
        date = pd.date_range(start= self.decision_d, periods= self.policy_plot.shape[0])
        fig, ax = plt.subplots(2, 1)
        df = pd.DataFrame(data = self.policy_plot, index = date, columns = ['Percent reduction in contacts through social distancing',\
                        'Testing capacity – maximum tests per day through contact tracing', \
                        'Testing capacity – maximum tests per day through universal testing'])
        df.plot(y ='Percent reduction in contacts through social distancing', \
               label = 'User entered decision choice for: \nPercent reduction in contacts through social distancing', \
               use_index = True, fontsize = 10, marker='.', ax = ax[0], c = 'k')
        df.plot(y ='Testing capacity – maximum tests per day through contact tracing', \
               label = 'User entered decision choice for: \nTesting capacity – maximum tests per day through contact tracing', \
               use_index = True, fontsize = 10, marker='.',ax = ax[1])
        df.plot(y ='Testing capacity – maximum tests per day through universal testing', \
               label = 'User entered decision choice for: \nTesting capacity – maximum tests per day through universal testing', \
               use_index = True, fontsize = 10, marker='.',ax = ax[1])
               
        ax[0].set_xlim(left = pd.Timestamp(self.start_d))
        ax[1].set_xlim(left = pd.Timestamp(self.start_d))
        ax[1].set_ylim(bottom = 0)
        ax[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        ax[0].set_ylabel('Proportion')
        ax[1].set_ylabel('Number of test')
        
        bbox = dict(boxstyle="round", fc="0.8")
        arrowprops = dict(arrowstyle = "->",connectionstyle = "angle,angleA=0,angleB=70,rad=5")
        offset = 72
        y1 = df.loc[self.decision_d]['Percent reduction in contacts through social distancing']
        ax[0].annotate('Start of decision making',(self.decision_d, y1),xytext=(0.5*offset, 0.5*offset), \
                        textcoords='offset points',bbox=bbox, arrowprops=arrowprops)
        y2 = df.loc[self.decision_d]['Testing capacity – maximum tests per day through contact tracing']
        ax[1].annotate('Start of decision making',(self.decision_d, y2),xytext=(0.5*offset, 0.3*offset), \
                       textcoords='offset points',bbox=bbox, arrowprops=arrowprops)

        fig.suptitle('User entered decision choice')

        plt.savefig('8.png',dpi = self.dpi)
        plt.close()
      
        return df



    """def plot_time_output(self):
        plt.style.use('seaborn')
        plt.plot(self.time_step, self.num_inf_plot, label = 'number of infected')
        plt.plot(self.time_step, self.num_hosp_plot, label = 'number of hospitalization')
        plt.plot(self.time_step, self.num_dead_plot, label = 'number of deaths')
        plt.xlabel('day')
        plt.ylabel('number')
        plt.legend()
        plt.savefig('time.png')
        plt.close()"""