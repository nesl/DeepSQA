import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
 
import keras
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
    pass


from sqa_data_gen.sqa_gen_engine import *
import numpy as np
from os import listdir
import pandas as pd


from sqa_data_gen.sqa_gen_engine import *
import json
import pickle

from sqa_data_gen.dataset_analysis import *

question_family_file = 'question_family.json'

datapath = 'source_dataset/opportunity'

_, label_list, _ = extract_data_from_file('S1-ADL5.dat',  datapath, 
                                         plot_option = False, 
                                         show_other = False)

label_list[0][0] = 'Do other activities'
label_list[0][1] = 'Open the front Door'
label_list[0][2] = 'Open the back Door'
label_list[0][3] = 'Close the front Door'
label_list[0][4] = 'Close the back Door'
label_list[0][5] = 'Open the Fridge'
label_list[0][6] = 'Close the Fridge'
label_list[0][7] = 'Open the Dishwasher'
label_list[0][8] = 'Close the Dishwasher'
label_list[0][9] = 'Open the first Drawer'
label_list[0][10] = 'Close the first Drawer'
label_list[0][11] = 'Open the second Drawer'
label_list[0][12] = 'Close the second Drawer'
label_list[0][13] = 'Open the third Drawer'
label_list[0][14] = 'Close the third Drawer'
label_list[0][15] = 'Clean the Table'
label_list[0][16] = 'Drink from the Cup'
label_list[0][17] = 'Toggle the Switch'


        
## specify the used data and generation parameters.
test_file_list = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                   'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat',
                   'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat',
                   'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
                  ]

stride = 400
window_size = 1500

save_file = 's1234_'+str(window_size) +'_'+str(stride)
save_folder = 'sqa_data'
save_model_folder = 'trained_models/'

generated_data = sqa_gen_engine(test_file_list,
                                datapath,
                                label_list,
                                data_split = 'Test', # either train or test
                                save_folder = save_folder,
                                save_model_folder = save_model_folder,
                                sqa_data_name = save_file,
                                window_size = window_size, stride = stride,
                                question_family_file = 'sqa_data_gen/question_family.json',
                                show_other = False  # not include "other" in activity list/scene_list
                                )

try: 
    print('The length of generated data is: ', len(generated_data['questions']) )
except:
    print('Loading json data...')
    with open(save_folder+ '/'+ save_file+'.json') as json_file:
        generated_data = json.load(json_file)
    print('Json data loaded: '+ save_folder+ '/'+ save_file+'.json')
    print('The length of generated data is: ', len(generated_data['questions']) )

pd_data = pd.DataFrame.from_dict(generated_data['questions'])

d_t = sqa_dataset(pd_data[(pd_data.question_family_index!=14) & (pd_data.question_family_index!=15)] )

exp_data_list =   ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                   'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat',
                   'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat',
                   'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
                  ]

exp_data_ind = pd_data.context_source_file.isin(exp_data_list)
exp_data_ind = exp_data_ind

d_t = sqa_dataset(pd_data[(pd_data.question_family_index!=14) & (pd_data.question_family_index!=15) & exp_data_ind]  )

print('Current datasize: ', d_t.size)
print('Need to compress to %.4f of original size.' % (150000/d_t.size) )
print(np.sqrt((150000/d_t.size)))


d_t.balance_question_dist(decrease_ratio = 0.29)
d_t.question_types()

d_t.balance_ans_dist( decrease_ratio = 0.29)
d_t.answer_distribution(topk=5)

print('The current data size: %d, taking memory %.2f GB. '%(d_t.size, d_t.size*1500*77*4/1024/1024/1024 ))

# change the saving name here
save_file = 's1234_'+str(window_size) +'_'+str(stride)

pickle_path = save_folder+'/'+save_file+ '_balanced' + '.pkl'

d_t.data.to_pickle(pickle_path)
print('Data saved: ',pickle_path)


load_pd_data = pd.read_pickle("sqa_data/s1234_1800_600_balanced.pkl")
d_t = sqa_dataset(load_pd_data )
print('Current datasize: ', d_t.size)
print('Need to compress to %.4f of original size.' % (200000/d_t.size) )
print(np.sqrt((200000/d_t.size)))


pickle_path = save_folder+'/'+'s1234_1800_600_balanced_small' + '.pkl'

d_t.data.to_pickle(pickle_path)
print('Data saved: ',pickle_path)

# # Extracting unique sensory scenes and store them
data_select_final = pd.DataFrame.from_dict(generated_data['questions'])
unique_scene = data_select_final.context_index.unique()

df = data_select_final[['context_source_file', 'context_start_point'] ].copy()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

def get_sensory_scene( file_name, startpoint, window_size ):
    
    data = []
    labels = []
    
    cols = [
    38, 39, 40, 41, 42, 43, 44, 45, 46, # InertialMeasurementUnit BACK
    51, 52, 53, 54, 55, 56, 57, 58, 59, # InertialMeasurementUnit RUA 
    64, 65, 66, 67, 68, 69, 70, 71, 72, # InertialMeasurementUnit RLA 
    77, 78, 79, 80, 81, 82, 83, 84, 85, # InertialMeasurementUnit LUA
    90, 91, 92, 93, 94, 95, 96, 97, 98, # InertialMeasurementUnit LLA
    103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116,  # IMU L+R shoe
    117, 118, 119, 120, 121, 122, 123, 124,
    125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 
    250] # last col is for mid-lvl label, not used here.
    cols = [x-1 for x in cols]
    
    with open('source_dataset/opportunity/%s' % file_name, 'r') as f:

        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            elem = []
            for ind in cols:
                elem.append(line[ind])
            if sum([x == 'NaN' for x in elem]) == 0:
                data.append([float(x) / 1000 for x in elem[:-1]])
    
    data_x = np.asarray(data)
    seg_data_x = data_x[startpoint:startpoint+window_size]
     
    return seg_data_x


context_dict = {}
context_ebd_dict = {}

for i in range(df.shape[0]):
    context_key = df['context_source_file'][i]+ '_' +str(df['context_start_point'][i] )
    print(i,': \t', context_key)
    
    current_data_i = get_sensory_scene( df['context_source_file'][i] , df['context_start_point'][i], window_size )
    context_dict[context_key] = current_data_i
    current_data_i = np.expand_dims(current_data_i, -1)
    current_data_i = np.expand_dims(current_data_i, 0)
    current_data_i = np.swapaxes(current_data_i,1,2)


context_data_path = save_folder+'/'+save_file+'_context.pkl'

context_data = {}
context_data['raw'] = context_dict
context_data['embedding'] = context_ebd_dict


with open(context_data_path, 'wb') as handle:
    pickle.dump(context_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Context Saved at: ", context_data_path)
