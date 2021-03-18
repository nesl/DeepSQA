import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class sqa_result():
    
    def __init__(self, pickle_data_path, baseline_result_path, mac_result_path = None ):
        self.pickle_data_path = pickle_data_path
        self.baseline_result_path = baseline_result_path
        self.mac_result_path = mac_result_path
        
        print('===========================================')
        print('============ Loading result ===========\n')
        print('Loading Pickle data from: ',pickle_data_path)
        self.load_pd_data = pd.read_pickle(pickle_data_path)
        
        print('Loading baseline results from: ', baseline_result_path )
        with open(baseline_result_path, 'rb') as handle:
            sim_result = pickle.load(handle)
        
        # whether mac data is available
        self.mac_flag = False
        if mac_result_path:
            print('Loading mac results from: ', mac_result_path )
            with open(mac_result_path, 'rb') as handle:
                mac_result = pickle.load(handle)
            self.mac_flag = True
            
        # need to integrate mac result...
        self.inference_result = sim_result['inference']
        self.valid_ind = sim_result['index']['valid']
        self.train_ind = sim_result['index']['train']
        self.learn_curves = sim_result['train_curve']
        
        if self.mac_flag:
            mac_learning_curve = {
                                    'val_loss': list(mac_result['history'][:,3]),
                                    'val_accuracy': list(mac_result['history'][:,2]),
                                    'loss': list(mac_result['history'][:,1]),
                                    'accuracy': list(mac_result['history'][:,0])
                                }
            self.learn_curves['mac'] = mac_learning_curve
            self.inference_result = np.concatenate([sim_result['inference'], 
                                                    np.expand_dims( mac_result['inference_result'], axis = 0)])
        
        self.prime_q_struct = None
        self.prime_data_ind = None
        
        print('\n============ Result loaded ===========')
        print('===========================================\n\n')
        
            
 
#     def get_prime_data(self):
#     # getting global prime data index.  (SWD: Test set might have only 1 answers: other answers in the training set)
#         uniq_q_struct = list(self.load_pd_data.question_structure.unique() )
#         self.prime_q_struct = []

#         for i in tqdm(uniq_q_struct):
#             ans_qstruct_i = list( self.load_pd_data[ self.load_pd_data.question_structure == i ].answer.unique())
#             if len(ans_qstruct_i)>1:
#                 self.prime_q_struct.append(i)
                
#         self.prime_data_ind = self.load_pd_data.question_structure.isin(self.prime_q_struct)
#         print('Original data size: %d, Prime data size: %d' %(self.prime_data_ind.shape[0], self.prime_data_ind.sum()))

    # get prime data for valid set only
    def get_prime_data(self):
    # getting valid prime data index. 
        uniq_q_struct = list(self.load_pd_data[self.valid_ind].question_structure.unique() )
        self.prime_q_struct = []

        for i in tqdm(uniq_q_struct):
            ans_qstruct_i = list( self.load_pd_data[ (self.valid_ind) & (self.load_pd_data.question_structure == i) ].answer.unique())
            if len(ans_qstruct_i)>1:
                self.prime_q_struct.append(i)
                
        self.prime_data_ind = self.load_pd_data.question_structure.isin(self.prime_q_struct)
        print('Original data size: %d, Prime data size: %d' %(self.valid_ind.sum(), (self.valid_ind & self.prime_data_ind).sum()))     
        
            
        
        
    def plot_learning_curve(self, draw_single = False, draw_comparison = True):
        
        train_curve = self.learn_curves
        
        if draw_single:
            print('\nLearning curves of each model: ')
            for model in train_curve.keys():        
                fig, ax1 = plt.subplots(figsize=(6,4))
                plt.grid()
                ax1.set_xlabel('Epoch (s)')
                ax1.set_ylabel('Training MAE / Loss')
                ax1.plot(train_curve[model]['loss'], '--', linewidth=3, color = 'C0', label = 'Loss: '+model)
                ax1.plot(train_curve[model]['val_loss'], '--',  linewidth=3, color = 'C1', label = 'Val_loss')
                ax1.legend(loc='center left')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Accuracy')  # we already handled the x-label with ax1
                ax2.plot(train_curve[model]['accuracy'], '-', linewidth=3, color = 'C2', label = 'Acc: '+model)
                ax2.plot(train_curve[model]['val_accuracy'], '-', linewidth=3, color = 'C3', label = 'Val_acc')
                ax2.legend(loc='center right')

                fig.tight_layout()            

        if draw_comparison:
            print('\nLearning performance comparison between models: ')
            color_alpha = 0.9
            fig = plt.figure(figsize=(6, 4))
            for model in train_curve.keys():        
                plt.plot(train_curve[model]['val_accuracy'], '-', linewidth=3, alpha=color_alpha, label = model)
            plt.grid()
            plt.xlabel('Epochs')
            plt.ylabel('Validation Acc')
            plt.legend()

        
        
    def plot_result(self, prime_data = False):
        """
        prime_data_ind: is the global index for all prime data.
        prime_data: is the option for selecting prime/non-prime data

        in the case of prime_data, return acc_list on prime data
        """
        ans_list_array = self.inference_result
        load_pd_data = self.load_pd_data
        test_ind = self.valid_ind 
        if prime_data:
            if self.prime_data_ind is None:
                self.get_prime_data()
            prime_data_ind = self.prime_data_ind
            
        

        # if plot the performance of prime set.
        if ( prime_data)  and (prime_data_ind is not None):
            print('Plotting the performance on prime dataset(removing questions with <2 possible Ans)...')   
            prime_test_ind = (test_ind&prime_data_ind)[test_ind]
            ans_list_array = ans_list_array[:,prime_test_ind]
            test_ind = test_ind&prime_data_ind
            print('Testing on Prime dataset with size of: ',ans_list_array.shape[1])
        elif ( prime_data)  and (prime_data_ind is None):
            print('The prime_data_ind not available, procceed with normal testing set.')
        # otherwise, dont care
        else:
            print('Plotting the performance on original dataset...')


        # ========== get testing data index ==========
        load_pd_data = load_pd_data[test_ind]

        # question type
        # q_type = [[0, 2, 3], [1, 4, 5, 13], [6, 7, 12, 14, 15], [10, 11], [8, 9]]  
        #['Exist', 'Count', 'Query', 'Compare', 'Compare-act']
        q_family_type_dict = {}

        q_family_type_dict[0] = 0
        q_family_type_dict[1] = 1
        q_family_type_dict[2] = 0
        q_family_type_dict[3] = 0
        q_family_type_dict[4] = 1
        q_family_type_dict[5] = 1
        q_family_type_dict[6] = 2
        q_family_type_dict[7] = 2
        q_family_type_dict[8] = 4
        q_family_type_dict[9] = 4
        q_family_type_dict[10] = 3
        q_family_type_dict[11] = 3
        q_family_type_dict[12] = 2
        q_family_type_dict[13] = 1

        type_list = []
        for q_i in load_pd_data['question_family_index']:
            type_list.append( q_family_type_dict[q_i] )
        type_list = np.array(type_list)

        # array([0, 1, 2, 3, 4])  5 types of questions, + overall performance
        ind_col = type_list
        cat_list = np.unique(type_list)


        ## ========== create accu_list for each category ==========
        acc_list = []
        # adding overall acc
        acc_list.append(ans_list_array.sum(axis = 1)/ans_list_array.shape[1])
        
        # adding binary and open question acc
        binary_family_list = [0, 2, 3, 8, 9, 10, 11]
        open_family_list = [1, 4, 5, 6, 7 , 12, 13, 14, 15]
        bin_ind = load_pd_data['question_family_index'].isin(binary_family_list)
        open_ind = load_pd_data['question_family_index'].isin(open_family_list)
        bin_acc = ans_list_array[:, bin_ind].sum(axis = 1)/sum(bin_ind)
        open_acc = ans_list_array[:, open_ind].sum(axis = 1)/sum(open_ind)
        acc_list.append(bin_acc)
        acc_list.append(open_acc)
        
        for i in cat_list:
            ind_i = (ind_col==i)
            ans_i = ans_list_array[:, ind_i]
            acc_i = ans_i.sum(axis =1 )/ans_i.shape[1]

            acc_list.append(acc_i)
        acc_list = np.array(acc_list)


        ## ========== plotting ==========
        x_axis = ['Overall', 'Binary', 'Open', 'Exist', 'Count', 'Query', 'Compare', 'Compare-act']
        x_pos = np.arange(len(x_axis))*10

        color_alpha = 0.75

        # Build the plot
        fig, ax = plt.subplots(figsize=(16,4))

    #     print(acc_list)
        # ACC figure
        ax.bar(x_pos-4, acc_list[:, 0] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos-3, acc_list[:, 1] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos-2, acc_list[:, 2] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos-1, acc_list[:, 3] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos+0, acc_list[:, 4] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos+1, acc_list[:, 5] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos+2, acc_list[:, 6] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos+3, acc_list[:, 7] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        if self.mac_flag:
            ax.bar(x_pos+4, acc_list[:, 8] , align='center', alpha=color_alpha, ecolor='black', capsize=10)

        ax.set_ylabel('Accuracy per question type')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_axis)
        ax.set_xlabel('Question Type')
        # ax.set_title('Size of training dataset available ')
        ax.yaxis.grid(True)

        if self.mac_flag:
            ans_legend = ['prior', 'prior_q', 'neural-symbolic', 'cnn', 'lstm',  'ConvLstm(mul)', 'ConvLstm(cat)',  'SAN', 'MAC']
        else: 
            ans_legend = ['prior', 'prior_q', 'neural-symbolic', 'cnn', 'lstm',  'ConvLstm(mul)', 'ConvLstm(cat)',  'SAN']

        plt.legend(ans_legend, bbox_to_anchor=(1.14,0), loc='lower right', fancybox=True, framealpha=0.95)
    #     plt.legend( loc="lower left")
    #     plt.legend(bbox_to_anchor=(1.04,1), loc="lower left")

        # # Save the figure and show
        # plt.tight_layout()
        # plt.ylim(0.67, 1)
        # plt.legend([sys_name, sys_name+' w/o sem.loss', 'CRNN', 'C3D'], loc='lower right', fancybox=True, framealpha=0.95)
        # plt.savefig('sim3-1.png', dpi=600)
        plt.show()    

        return acc_list
    
    

    def plot_result_q_family(self, prime_data = False):
        """
        """        
        ans_list_array = self.inference_result
        load_pd_data = self.load_pd_data
        test_ind = self.valid_ind 
        if prime_data:
            if self.prime_data_ind is None:
                self.get_prime_data()
            prime_data_ind = self.prime_data_ind
        
        # if plot the performance of prime set.
        if ( prime_data)  and (prime_data_ind is not None):
            print('Plotting the performance on prime dataset(removing questions with <2 possible Ans)...')   
            prime_test_ind = (test_ind&prime_data_ind)[test_ind]
            ans_list_array = ans_list_array[:,prime_test_ind]
            test_ind = test_ind&prime_data_ind
            print('Testing on Prime dataset with size of: ',ans_list_array.shape[1])
        elif ( prime_data)  and (prime_data_ind is None):
            print('The prime_data_ind not available, procceed with normal testing set.')
        # otherwise, dont care
        else:
            print('Plotting the performance on original dataset...')


        # ========== get testing data index ==========
        load_pd_data = load_pd_data[test_ind]
        if self.mac_flag:
            ans_legend = ['prior', 'prior_q', 'neural-symbolic', 'cnn', 'lstm',  'ConvLstm(mul)', 'ConvLstm(cat)',  'SAN', 'MAC']
        else: 
            ans_legend = ['prior', 'prior_q', 'neural-symbolic', 'cnn', 'lstm',  'ConvLstm(mul)', 'ConvLstm(cat)',  'SAN']


    #     # ans_list_array = np.array(ans_list)
    #     ans_list_array = result_test

        # ans_list_array = np.squeeze(ans_list_array, axis = 2)
        # print(ans_list_array.shape)

        # ==== question family ====
        # pd.unique(load_pd_data.question_family_index)
        # array([ 0,  1,  6,  7,  8,  9, 12, 13, 10, 11,  2,  4,  3,  5])
        ind_col = np.array(load_pd_data.question_family_index)
        cat_list = np.arange(14)

        acc_list = []

        for i in cat_list:
            ind_i = (ind_col==i)
            ans_i = ans_list_array[:, ind_i]
            acc_i = ans_i.sum(axis =1 )/ans_i.shape[1]

            acc_list.append(acc_i)
        #     break

        acc_list = np.array(acc_list)


        x_axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ]
        x_pos = np.arange(len(x_axis))*10

        color_alpha = 0.75

        # Build the plot
        fig, ax = plt.subplots(figsize=(20,4))

        # ACC figure
        ax.bar(x_pos-4, acc_list[:, 0] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos-3, acc_list[:, 1] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos-2, acc_list[:, 2] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos-1, acc_list[:, 3] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos-0, acc_list[:, 4] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos+1, acc_list[:, 5] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos+2, acc_list[:, 6] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        ax.bar(x_pos+3, acc_list[:, 7] , align='center', alpha=color_alpha, ecolor='black', capsize=10)
        if self.mac_flag:
            ax.bar(x_pos+4, acc_list[:, 8] , align='center', alpha=color_alpha, ecolor='black', capsize=10)


        ax.set_ylabel('Accuracy per question type')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_axis)
        ax.set_xlabel('Question Family')
        # ax.set_title('Size of training dataset available ')
        ax.yaxis.grid(True)

        plt.legend(ans_legend, loc='lower right', fancybox=True, framealpha=0.95)

        # # Save the figure and show
        # plt.tight_layout()

        # plt.ylim(0.67, 1)
        # plt.legend([sys_name, sys_name+' w/o sem.loss', 'CRNN', 'C3D'], loc='lower right', fancybox=True, framealpha=0.95)
        # plt.savefig('sim3-1.png', dpi=600)
        plt.show()

        return acc_list