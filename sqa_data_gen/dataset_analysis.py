import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import time

from sklearn.utils import resample

class sqa_dataset():
    
    """
    Input: pdd_data is the data frames of generated_data['questions']
    Output: stored values of all statisitcs.
    
    """
    
    ## global values:
    # Question family info
    q_info_mat = []
    q_id = np.arange(16)
    q_type = [0, 1, 0, 0, 1, 1, 2, 2, 4, 4, 3, 3, 2, 1, 2, 2] # encoding: 0-existence, 1-counting, 2-query(act+time), 3-comparison(integer), 4-comparison(action)
    q_a_type = [0, 1, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 2, 1, 3, 3] # encoding: 0-existence, 1-counting, 2-query(act), 3-query time
    q_steps = [2, 2, 3, 5, 3, 5, 3, 5, 7, 7, 7, 5, 3, 3, 2, 3] # of logic steps used.
    q_info_mat = np.array( [q_id, q_type, q_a_type, q_steps] ) 
    
    
    def __init__(self, pd_data):
        self.data = pd_data
        self.size = pd_data.shape[0]
    
    def basic_stats(self):
        # ===== basic stats of generated data =====
    
        print('\n==== Basic Stats ====') 
        print('The total number of generated questions: ',self.size)
        print('Used questions types: ', sorted(self.data.question_family_index.unique()) )

        # unique questions
        unique_questions = self.data['question'].unique()
        print('Number of generated unique questions: ',unique_questions.shape[0] )
        print('Number of unique questions struct: ',self.data['question_structure'].unique().shape[0] )

        # The unique number of scene used:
        unique_scene = self.data['context_index'].unique()
        print('Number of unique scene: ',unique_scene.shape[0] )
        
        # The number of valid Neural-symbolic answers
        oracle_ind = self.data['pred_answer']!='Invalid'  # Take the invalid answers into account...
        print('Number/percentage of valid Oracle answers: %d / (%.2f %%)' %(sum(oracle_ind), sum(oracle_ind)/self.size*100) )
        print('Oracle answers Accuracy: %.2f %%' %(sum(self.data.answer == self.data.pred_answer) /self.size*100))
        
        
    def question_length(self):
        # =========================================
        # question length distribution

        print('\n==== Question Length ====') 
        length_list = self.data['question'].str.split().str.len()
        # swd: need to change this length function later. 
        # this is inconsistent with the length func in "prepare_data", that one take '?'',' into account.
        
        print('The qeustion Length: ')
        print('Min: %d, Mean: %d, Max: %d'%(length_list.min(),length_list.mean(),length_list.max()))

        len_table = np.zeros([2, length_list.max()-length_list.min()+1])
        for i in range(len_table.shape[1]):
            len_table[:, i] = [i+length_list.min(), sum(length_list == (length_list.min()+i))]
        len_table[1,:] = len_table[1,:] /self.size #change number to frequence

#         # set numpy printing format 
#         import contextlib
#         @contextlib.contextmanager
#         def printoptions(*args, **kwargs):
#             original = np.get_printoptions()
#             np.set_printoptions(*args, **kwargs)
#             try:
#                 yield
#             finally: 
#                 np.set_printoptions(**original)
    #     with printoptions(precision=3, suppress=True):
    #         print(len_table)

        ### plot the question length distribution ####
        names = ['VQA', 'V7W', 'CLEVR-HUMAN', 'CLEVR', 'GQA']

        len_num = np.arange(2,14,1)
        len_freq = np.array([0, 0.034, 0.1448, 0.4156, 0.1683, 0.1128, 0.0688, 0.0271, 0.0109, 0.0054, 0, 0 ])
        len_num = np.expand_dims(len_num, -1)
        len_freq = np.expand_dims(len_freq, -1)
        len_q1 = np.concatenate((len_num, len_freq), axis =1)

        len_num = np.arange(3,16,1)
        len_freq = np.array([0.0241, 0.1218, 0.2636, 0.2129, 0.1472, 0.1062, 0.0483, 0.0259, 0.0126, 0.0073, 0.0053, 0.0033, 0 ])
        len_num = np.expand_dims(len_num, -1)
        len_freq = np.expand_dims(len_freq, -1)
        len_q2 = np.concatenate((len_num, len_freq), axis =1)

        len_num = np.arange(4,31,1)
        len_freq = np.array([0.0119, 0.1307, 0.2071, 0.1188, 0.1022, 0.0876, 0.0650, 0.0617, 0.0518, 0.0385, 
                             0.0206, 0.0192, 0.0119, 0.008, 0.0053, 0.0023, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015
                            , 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015])
        len_num = np.expand_dims(len_num, -1)
        len_freq = np.expand_dims(len_freq, -1)
        len_q3 = np.concatenate((len_num, len_freq), axis =1)

        len_num = np.arange(4,31,1) #???? error 
        len_freq = np.array([0, 0.003, 0.0166, 0.0212, 0.0133, 0.0119, 0.0146, 0.0239, 0.0418, 0.0690, 0.0863, 
                             0.0803, 0.0604, 0.0478, 0.0531, 0.0531, 0.0465, 0.0398, 0.0372, 0.0365, 0.0332, 
                             0.0305, 0.0272, 0.0232, 0.0199, 0.0179, 0.0159])
        len_num = np.expand_dims(len_num, -1)
        len_freq = np.expand_dims(len_freq, -1)
        len_q4 = np.concatenate((len_num, len_freq), axis =1)

        len_num = np.arange(3,27,1)
        len_freq = np.array([ 0.0086, 0.0757, 0.1301, 0.1354, 0.0976, 0.0889, 0.1201, 0.0816, 0.0903, 0.0398, 
                             0.0465, 0.0265, 0.0119, 0.01, 0.0074, 0.006, 0.004, 0.002, 0.002, 0.002, 0.002
                            , 0.002, 0.002, 0.002])
        len_num = np.expand_dims(len_num, -1)
        len_freq = np.expand_dims(len_freq, -1)
        len_q5 = np.concatenate((len_num, len_freq), axis =1)

        fig= plt.figure(figsize=(16, 4))
        plt.plot(len_q1[:,0],len_q1[:,1], '-', linewidth=2)
        plt.plot(len_q2[:,0],len_q2[:,1], '-', linewidth=2)
        plt.plot(len_q3[:,0],len_q3[:,1], '-', linewidth=2)
        plt.plot(len_q4[:,0],len_q4[:,1], '-', linewidth=2)
        plt.plot(len_q5[:,0],len_q5[:,1], '-', linewidth=2)

        plt.plot(len_table[0,:],len_table[1,:], '-', linewidth=4)
        plt.grid()
        plt.ylabel('Frequency')
        plt.xlabel('Question Length')
        plt.title('Question length distribution')
        plt.legend(names + ['SQA'])
        
#         return [len_q1, len_q2, len_q3, len_q4, len_q5, len_table]
        
        
        
    def question_types(self):
        print('\n==== Question Family ====') 
    
        fam_list = self.data['question_family_index']
        uniq_family = sorted(fam_list.unique())

        fam_table = np.zeros([2, len( uniq_family)])
        for idx, fam_i in enumerate(uniq_family):
            fam_table[:, idx] = [fam_i, sum(fam_list == fam_i)]
        fam_table[1,:] = fam_table[1,:] /self.data.shape[0]  # table of freq instead of numbsers

        ## calculating the distribution frequencies
        #5 types of questionss 0-existence, 1-counting, 2-query(act+time), 3-comparison(integer), 4-comparison(action)
        fam_type_table = np.zeros([2, 5]) 
        for i in range(5): 
            type_id = self.q_info_mat[0, self.q_info_mat[1,: ]==i]
            fam_table_ind = np.zeros_like(fam_table[0,:], dtype = bool)
            for type_id_j in type_id:
                fam_table_ind = (fam_table_ind | (fam_table[0,:] == type_id_j) )
            fam_type_table[:, i] = [i, fam_table[1, fam_table_ind].sum() ]

        #4 types of questions answers:   encoding: 0-existence, 1-counting, 2-query(act), 3-query time
        ans_type_table = np.zeros([2, 4]) 
        for i in range(4): 
            type_id = self.q_info_mat[0, self.q_info_mat[2,: ]==i]
            ans_table_ind = np.zeros_like(fam_table[0,:], dtype = bool)
            for type_id_j in type_id:
                ans_table_ind = (ans_table_ind | (fam_table[0,:] == type_id_j) )
            ans_type_table[:, i] = [i, fam_table[1, ans_table_ind].sum() ]



        # Bar plot for question family distribution & question type distribution & Question answer type distribution    
        plt.figure(figsize=(5,8))

        x_axis = np.arange(fam_table.shape[1])
        x_pos = np.arange(len(x_axis))*1
    #     fig, ax = plt.subplots(figsize=(9,6))
        ax1 = plt.subplot(311)
        ax1.bar(x_pos, fam_table[1,:] , align='center', alpha=0.5, ecolor='black', capsize=10)
        ax1.set_ylabel('Frequencies')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_axis)
        ax1.set_title('Question families ')
        ax1.yaxis.grid(True)

        x_axis = ['Exist', 'Count', 'Query', 'Compare', 'Compare-act']
        x_pos = np.arange(len(x_axis))*1
        ax2 = plt.subplot(312)
        ax2.bar(x_pos, fam_type_table[1,:] , align='center', alpha=0.5, ecolor='blue', capsize=10)
    #     ax2.set_ylabel('Frequencies of questions per family')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_axis)
        ax2.set_ylabel('Frequencies')
        ax2.set_title('Question Types ')
        ax2.yaxis.grid(True)

        x_axis = ['Binary', 'Count', 'Query', 'QueryTime']
        x_pos = np.arange(len(x_axis))*1
        ax3 = plt.subplot(313)
        ax3.bar(x_pos, ans_type_table[1,:] , align='center', alpha=0.5, ecolor='red', capsize=10)
    #     ax3.set_ylabel('Frequencies of questions per family')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x_axis)
        ax3.set_ylabel('Frequencies')
        ax3.set_title('Answer Types ')
        ax3.yaxis.grid(True)
        plt.show()
        
        # plot q type distribution: Pie chart
        # where the slices will be ordered and plotted counter-clockwise:
        print('==== Question Type distribution ====') 
        labels = ['Exist', 'Count', 'Query', 'Compare', 'Compare-act']
        
        fig1, ax1 = plt.subplots()
        ax1.pie(fam_type_table[1,:], labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        fig = plt.gcf()
        fig.set_size_inches(4,4) # or (4,4) or (5,5) or whatever
        plt.show()
#         return fam_type_table[1,:]
        
        
        
        
    def logic_steps(self):
        print('\n==== Number of semantic steps ====') 
        
        fam_list = self.data['question_family_index']
        uniq_family = sorted(fam_list.unique())
        fam_table = np.zeros([2, len( uniq_family)])
        for idx, fam_i in enumerate(uniq_family):
            fam_table[:, idx] = [fam_i, sum(fam_list == fam_i)]
        fam_table[1,:] = fam_table[1,:] /self.data.shape[0]  # table of freq instead of numbsers
        
        fam_step_table = np.zeros([2, 4]) 

        for idx, i in enumerate([2, 3, 5, 7]):  # only 4 kinds of different steps 
            type_id = self.q_info_mat[0, self.q_info_mat[3,: ]==i]
            fam_table_ind = np.zeros_like(fam_table[0,:], dtype = bool)
            for type_id_j in type_id:
                fam_table_ind = (fam_table_ind | (fam_table[0,:] == type_id_j) )
            fam_step_table[:, idx] = [i, fam_table[1, fam_table_ind].sum() ]
        
        # plot q type distribution:
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = ['2', '3', '5', '7']
        sizes = fam_step_table[1,:]
        
        plt.figure(figsize=(18, 4))  # not working? cannot change size of fig...
        ax1 = plt.subplot(121)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig = plt.gcf()
        fig.set_size_inches(7,7) # or (4,4) or (5,5) or whatever
        
        ax2 = plt.subplot(122)
        x_axis = labels
        x_pos = np.arange(len(x_axis))*1
        ax2.bar(x_pos, sizes , align='center', alpha=0.5, ecolor='red', capsize=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_axis)
        ax2.set_ylabel('Frequencies')
        ax2.set_title('Answer Types ')
        ax2.yaxis.grid(True)
        plt.show()

        print('The average number of semantic steps is: ', sum( fam_step_table[0,:]*fam_step_table[1,:] ))


    def answer_distribution(self, topk = 5):
        # =========================================
        # Answer distribution
        print('\n==== Question Answers ====') 
        pd_data_classify = self.data[(self.data.question_family_index!=14) & (self.data.question_family_index!=15)]
        unique_answer = pd_data_classify['answer'].unique()
        print('Unique answer number: ', unique_answer.shape[0])
        print('\nUnique classification answer: (w/o 14&15)')
        print(unique_answer)   

        print('\nAnswer distribution for all question types: ')
        print('[Exist, Count, Query, Compare, Compare-act]')   
        #     [0 2 3]
        #     [ 1  4  5 13]
        #     [ 6  7 12 14 15]
        #     [10 11]
        #     [8 9]
        fig= plt.figure(figsize=(18, 4))
        plt.subplot(151)
        self.data[(self.data.question_family_index ==0) | 
            (self.data.question_family_index ==2)| 
            (self.data.question_family_index ==3)]['answer'].value_counts().plot(kind='pie')
        plt.subplot(152)
        self.data[(self.data.question_family_index ==1) | 
            (self.data.question_family_index ==4)| 
            (self.data.question_family_index ==5)| 
            (self.data.question_family_index ==13)]['answer'].value_counts()[:10].plot(kind='pie')
        plt.subplot(153)
        self.data[(self.data.question_family_index ==6) | 
            (self.data.question_family_index ==7)| 
    #         (pd_data.question_family_index ==14)|   # not classification
    #         (pd_data.question_family_index ==15)| 
            (self.data.question_family_index ==12)]['answer'].value_counts().plot(kind='pie')
        plt.subplot(154)
        self.data[(self.data.question_family_index ==10) | 
            (self.data.question_family_index ==11)]['answer'].value_counts().plot(kind='pie')
        plt.subplot(155)
        self.data[(self.data.question_family_index ==8) | 
            (self.data.question_family_index ==9)]['answer'].value_counts().plot(kind='pie')
        plt.show()
        
        if topk == 0:
            return
        from tabulate import tabulate # print answer distribution for some questions
        all_question_struct = list(pd_data_classify['question_structure'].value_counts().index)
        print('\nAnswer distribution top-%d questions: '%topk)
        for struct_i in all_question_struct[0:topk]:
            struct_i_data = pd_data_classify[pd_data_classify['question_structure']==struct_i]['answer'].value_counts() 
            print('\n',struct_i)
            print(tabulate([struct_i_data], headers=struct_i_data.index, tablefmt='orgtbl'))
        
        
        
        
        
        
    def context_question(self):
        # context distribution
        print('\n==== Info about source data and QA ====') 
        
        unique_scene = self.data['context_index'].unique()

        print('Number of unique scene: \n',unique_scene.shape )

        # Scene distribution among source data files
        unique_source_file = self.data['context_source_file'].unique()

        file_list = []
        contxt_num_list = []
        q_num_list = []

        for file_name in unique_source_file:
            data_i = self.data[self.data['context_source_file'] ==file_name]
            context_number_i = data_i['context_index'].unique().shape[0]
            question_number_i = data_i.shape[0]
            contxt_num_list.append(context_number_i)
            q_num_list.append(question_number_i)
            file_list.append(file_name)

        df = pd.DataFrame({'file_list': file_list, 
                           'contxt_num_list': contxt_num_list,
                           'q_num_list': q_num_list})

        x_axis = file_list
        x_pos = np.arange(len(x_axis))*1

        fig = plt.figure(figsize=(5,3)) # Create matplotlib figure

        ax = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

        width = 0.3

        df.contxt_num_list.plot(kind='bar', alpha=0.5,color='red', ax=ax, width=width, position=1)
        df.q_num_list.plot(kind='bar', alpha=0.5,color='blue', ax=ax2, width=width, position=0)

        ax.set_ylabel('Context_num')
        ax2.set_ylabel('Question_num')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_axis)
        ax.legend(['Cxt Num'], loc='upper right', bbox_to_anchor=(1, 0.9))
        ax2.legend(['Que Num'], loc='upper right', bbox_to_anchor=(1, 1))

        plt.show()

        # average questions per scene
        print('Avg num of questions per scene : \n',self.data['question_index'].shape[0]/unique_scene.shape[0])
        

        

    def balance_ans_dist(self, decrease_ratio = 0.5):
        """
        adjust the answer distribution by reducing the amout of redundant training data
        Iterative process...
        """

        q_type = [[0, 2, 3], [1, 4, 5, 13], [6, 7, 12, 14, 15], [10, 11], [8, 9]]
        # q_type for 0, 1, 2, 3, 4 
        
        type_data_list = []
        
        for type_i in q_type:
            print('Question Family: ', type_i)
            # get type_i data
            type_i_data = self.data[ self.data.question_family_index.isin( type_i ) ]
            org_size = len(type_i_data)
            
            while len(type_i_data) > decrease_ratio * org_size:
                # balance answer dist
                ans_dist = type_i_data.answer.value_counts()
                ans_list = ans_dist.index
                ans_freq = list(ans_dist)

                df_majority1 = type_i_data[type_i_data['answer']==ans_list[0]]
                df_other = type_i_data[type_i_data['answer']!=ans_list[0]]

                maj_class1 = resample(df_majority1, 
                                     replace=False,     
                                     n_samples= int(ans_freq[0] * 0.8),    
                                     random_state=123) 

                type_i_data = pd.concat([maj_class1,df_other])
            
            type_data_list.append(type_i_data)
            
        self.data = pd.concat(type_data_list)
        self.size = len(self.data)
                
            
            
        
        
    def balance_question_dist(self, decrease_ratio = 0.5):
        """
        adjust the question type distribution by reducing the amout of redundant training data  
        """
        q_type = [[0, 2, 3], [1, 4, 5, 13], [6, 7, 12, 14, 15], [10, 11], [8, 9]]
        # q_type for 0, 1, 2, 3, 4 
        
        org_size = len(self.data)
        
        while len(self.data) > decrease_ratio * org_size:
            print('Progresss: ', len(self.data)/org_size)
            
            
            freq_list = [len(self.data[self.data.question_family_index.isin(i)]) for i in q_type]
            max_ind = freq_list.index(max(freq_list))  
            data_max_ind = self.data.question_family_index.isin( q_type[max_ind] )
            df_majority1 = self.data[data_max_ind]
            df_other = self.data[~data_max_ind]

            
            maj_class1 = resample(df_majority1, 
                                 replace=False,     
                                 n_samples= int(len(df_majority1) * 0.8),    
                                 random_state=123) 
            
            self.data = pd.concat([maj_class1,df_other])
            self.size = len(self.data)
            
            
    
    # question_visualization....
    def scan_question(self, q_family_list = list(range(16)), stop_time = 1):
        """
        """
        scan_ind = self.data.question_family_index == -1
        print('Scan question families: ', q_family_list)
        for qf_i in q_family_list:
            scan_ind = scan_ind | (self.data.question_family_index == qf_i)
        
        for quesstion_i in self.data[scan_ind].question:
            print(quesstion_i, end='\r')
            time.sleep(stop_time)
            print(' '*1000, end='\r')
        #     break
        return
        
            
            
        
