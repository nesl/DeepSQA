import numpy as np
from sqa_data_gen.data_extraction import *
from os import listdir
import pandas as pd

from sqa_data_gen.question_generation import *
from sqa_data_gen.synonym_change import synonym_change

import json

from datetime import date
from tqdm import tqdm

from keras.models import load_model 


def sqa_gen_engine(file_list,
                   datapath,
                   label_list,
                   data_split, # either train or test
                   save_folder,
                   save_model_folder,
                   sqa_data_name,
                   window_size = 1800, stride = 900,
                   question_family_file = 'question_family.json',
                   source_dataset = 'opp',
                   show_other = False
                   ):
    
    """

    """
    # loading pretrained source model to get primary events.
    if source_dataset == 'opp':
        model_name = 'single_1'
        save_path = save_model_folder+ 'opp_model/'+ model_name+'.hdf5'
        trained_model_1 = load_model(save_path)

        model_name = 'single_2'
        save_path = save_model_folder + 'opp_model/'+ model_name+'.hdf5'
        trained_model_2 = load_model(save_path)
    elif source_dataset == 'es':
        model_name = 'naive_classifier'
        save_path = save_model_folder + 'es_model/'+ model_name+'.hdf5'
        trained_model_1 = load_model(save_path)
    else:
        print('==== wrong source dataset type! ====')
        return

    window_size = window_size
    stride = stride
    file_list = file_list
    # data_folder = 'dataset'
    # # file_list = [ i for i in listdir(data_folder) if '.dat' in i]
    # file_list = ['S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']
    
    today = date.today()
    # mm/dd/y
    date_str = today.strftime("%m/%d/%y")
    
    # structure of question family
    gen_sqa_data = {}
    gen_sqa_data['info'] = {}
    gen_sqa_data['questions'] = []

    gen_sqa_data['info']['date'] = date_str
    gen_sqa_data['info']['license'] = 'Creative Commons Attribution (CC BY 4.0)'
    gen_sqa_data['info']['split'] = data_split
    gen_sqa_data['info']['version'] = '1.0'


    context_counter = 0


    for source_file_i in file_list:
        label_y, _, data_x = extract_data_from_file(source_file_i, 
                                                    datapath = datapath,
                                                    plot_option = False, 
                                                    show_other = show_other, 
                                                    source_data = source_dataset,
                                                    )
        ## Whether using "other" as one activity? No ... (3 places)
        
        print('Extracting %s file....'%(source_file_i)) 

        # generate context and questions using sliding window
        for startpoint_i in tqdm(range(0, data_x.shape[0]-window_size, stride)):

            # the sampling rate for opportunity is 30HZ, window is 60s
            seg_x ,seg_y_list, startpoint = data_split_tools(data_x, 
                                                 label_y, 
                                                 window_size, 
                                                 startpoint = startpoint_i)
            # visualize_data_labels(seg_y_list ,label_list, show_other = True)
            
            
            
            scene_list_1 = series2graph(seg_y_list[0], label_list[0], show_graph = False, show_other = show_other)
            scene_list_2 = series2graph(seg_y_list[1], label_list[1], show_graph = False)
            scene_lists = [scene_list_1, scene_list_2]
            
            # =====  get predicted scene_list using pre-trained source classifier.=====
            if source_dataset == 'opp':
                # need to reshape the input X
                seg_x = np.expand_dims(seg_x, axis=-1)
                seg_x = np.expand_dims(seg_x, axis=-1)
                
                seg_y_list_pred = [np.argmax(trained_model_1.predict(seg_x), axis=1), 
                                   np.argmax(trained_model_2.predict(seg_x), axis=1)]
                scene_list_1_pred = series2graph(seg_y_list_pred[0], label_list[0], show_graph = False, show_other = show_other)
                scene_list_2_pred = series2graph(seg_y_list_pred[1], label_list[1], show_graph = False)
                scene_lists_pred = [scene_list_1_pred, scene_list_2_pred]
            
            if source_dataset == 'es': 
                
                seg_y_list_pred = [np.argmax(trained_model_1.predict(seg_x), axis=1),]
                scene_list_1_pred = series2graph(seg_y_list_pred[0], label_list[0], show_graph = False, show_other = show_other)
                scene_lists_pred = [scene_list_1_pred, scene_list_1_pred]
            
            
            # modify question generator: it takes 2 sets of scene_list (real and predicted), and 2 answers.
            question_family_index, question_nl, answer_nl, answer_nl_p, question_struct = question_generator(scene_lists, 
                                                                                                             scene_lists_pred,
                                                                                                             question_family_file,
                                                                                                             label_list,
                                                                                                             show_other = show_other,
                                                                                                             question_validation = True,
                                                                                                             source_data = 'es',
                                                                                                             diagnose = False)

            for new_qf_ind, new_q_i, new_ans_i, new_ans_i_p, new_q_struct_i in zip(question_family_index, question_nl, answer_nl, answer_nl_p, question_struct):

                new_question_data = {}
                new_question_data['context_source_file'] = source_file_i
                new_question_data['context_start_point'] = startpoint
                new_question_data['context_index'] = context_counter
                # adding question variations by changing words to synonyms
                new_question_data['question'] = synonym_change(new_q_i) 
                new_question_data['answer'] = new_ans_i
                new_question_data['pred_answer'] = new_ans_i_p # predicted answer using neural symbolic appraoch
                new_question_data['question_family_index'] = new_qf_ind
                new_question_data['question_structure'] = new_q_struct_i
                new_question_data['question_index'] = 0 # need to get modified later
                new_question_data['split'] = data_split

                gen_sqa_data['questions'].append(new_question_data)

            context_counter += 1


    print('The total number of generated questions: ',len(gen_sqa_data['questions']))



    # Modify the question index:
    for q_idx, q_i in enumerate(gen_sqa_data['questions']):
        q_i['question_index'] = q_idx



    # saving generated questions

    save_file = sqa_data_name+'.json'
    save_folder = save_folder
    save_path = save_folder+'/'+save_file

    with open(save_path, 'w') as outfile:
        json.dump(gen_sqa_data, outfile)
        
    return gen_sqa_data