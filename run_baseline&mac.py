import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
 
# Do other imports now...
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')



##============= Run baselines ==============
# import numpy as np
# from sqa_models.run_baselines import *
# print('\n==== Run baselines ====\n')



# processed_saving_folder = 'sqa_data/'
# processed_data_path = 'sqa_data/opp_sim4_split.npz'   #================
# pickle_data_path = processed_saving_folder + 's1234_1800_600_balanced_small.pkl'

# dl_model_save_folder = 'opp_sim4_new/'
# epochs_num = 40
# result_name = 'opp_sim4_new'



# print('processed data path: ', processed_data_path)
# print('pickle data path: ', pickle_data_path)
# print('epochs: ', epochs_num)
# print('save dl models: ', dl_model_save_folder)
# print('save results: ', result_name)


# run_baselines(processed_data_path, 
#                   pickle_data_path,
#               dl_model_save_folder,
#                   epochs_num,
#               result_name,
#               train_dl = True,
#               source_data = 'opp'
#                      )




# ============= Run MAC ==============
import numpy as np
from sqa_models.run_mac import *
print('\n==== Run MAC ====\n')

dataset_path = 'sqa_data/opp_sim5_split.npz'   #================
hyper_parameters = {
    'n_words': 400001,
    'dim': 512,
    'glove_embeding': False,
    'ebd_train': True,
    'n_answers': 27,  # 13 for es, 27 for opp  ================
    'dropout': 0.15,
    'batch_size': 64,  # 32 for es, 64 for opp  ================
    'learning_rate': 1e-4,
    'weight_decay': 1e-4, 
}
epochs = 80  # 20 for es, 40 for opp  ================
model_save_folder = 'trained_models/opp_sim5/'   #================
result_save_name = 'result/opp_sim5_mac.pkl'   #================


print('processed data path: ', dataset_path)
print('pickle data path: N/A')
print('epochs: ', epochs)
print('save dl models: ', model_save_folder)
print('save results: ', result_save_name)


run_mac_model(dataset_path, 
              hyper_parameters,
              epochs,
              model_save_folder,
              result_save_name,
              source_data = 'opp')  #================