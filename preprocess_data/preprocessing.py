# def preprocess_data:

import preprocess_data.embedding as ebd
import preprocess_data.prepare_data as prepare_data
import os
import json
import pandas as pd
import os
import numpy as np

def dataset_split(pd_data, test_list):
    """
    """
    
    data_select_index_test = pd_data['context_source_file']=='000'

    for test_file_i in test_list:
        data_select_index_test =  (pd_data['context_source_file']==test_file_i)| (data_select_index_test)

    print('Testing data percentage: ', sum(data_select_index_test)/pd_data.shape[0])
    print('Testing data number: ', sum(data_select_index_test) )
    print('Testing data unique scene: ', len(pd_data[data_select_index_test].context_index.unique()) )

    return data_select_index_test

def preprocess_data(data_name, 
                    save_name,
                    data_folder = 'sqa_data/', 
                    create_ebd = False,
                    glove_path = 'glove/glove.6B.300d.txt',
                    context_name = 's1234_1800_600_context.pkl',
                    source_data = 'opp',
                    win_len = 1800
                        ):
    
    
    # load embd matrix and word index
    if create_ebd:
        print('Creating embedding matrix.')
        ebd.create(glove_path)
        
    word_idx = ebd.load_idx()
    embedding_matrix = ebd.load()
    print('Shape of embedding matrix:', embedding_matrix.shape)
    
    
    
    # loading balanced data pkl file, and context data
    test_data_path = data_folder + data_name
    context_data_path = data_folder + context_name
    print('Loading pickle data from: ',test_data_path)
    print('Loading context data from: ',context_data_path)


    # preparing data
    print('Loading questions ...')
    question_matrix = prepare_data.get_questions_matrix(test_data_path, source_data)
    print('Q data shape:', question_matrix.shape)

    print('Loading answers ...')
    answer_matrix = prepare_data.get_answers_matrix(test_data_path, source_data)
    print('A data shape:', answer_matrix.shape)

    print('Loading sensory context ...')
    sensory_matrix = prepare_data.get_sensory_context(test_data_path, context_data_path,  source_data, win_len, embedding = False)  
    print('S data shape:', sensory_matrix.shape)
    ### adjusting the shape of raw sensory data
    # sensory_matrix = np.expand_dims(sensory_matrix, axis=-1)
    # sensory_matrix = np.swapaxes(sensory_matrix,1,2)
    # print(sensory_matrix.shape)
    if source_data == 'opp':
        print('Mem used (for S matrix): %.2f GB.' %(sensory_matrix.shape[0]*1800*77*4/1024/1024/1024) )  # gb used
    else:
        print('Mem used (for S matrix): %.2f GB.' %(sensory_matrix.shape[0]*200*225*4/1024/1024/1024) )  # gb used

    
    ## normalizing dataset? :  for raw data...NO not helping
    # data_min = sensory_matrix.min(axis = 0)
    # data_max = sensory_matrix.max(axis = 0)
    # sensory_matrix = 2*(sensory_matrix - data_min)/(data_max - data_min) -1
    # print(sensory_matrix.max(), sensory_matrix.min() , sensory_matrix.mean())
    
    
    
    # loading generated questions pickle
    pd_data = pd.read_pickle(test_data_path)
    
    
    ### splitting method 1: based on context
    if source_data == 'opp':
        valid_list = [  'S1-ADL1.dat', 'S2-ADL1.dat', 'S3-ADL1.dat', 'S4-ADL1.dat',
                        'S1-ADL3.dat', 'S2-ADL3.dat', 'S3-ADL3.dat', 'S4-ADL3.dat',
                        'S1-ADL2.dat', 'S2-ADL2.dat', 'S3-ADL2.dat', 'S4-ADL2.dat',
                     ]

        train_list = [ 
                      'S1-ADL4.dat', 'S2-ADL4.dat', 'S3-ADL4.dat', 'S4-ADL4.dat',
                      'S1-ADL5.dat', 'S2-ADL5.dat', 'S3-ADL5.dat', 'S4-ADL5.dat',
                      'S1-Drill.dat', 'S2-Drill.dat', 'S3-Drill.dat', 'S4-Drill.dat',
                    ]
    else: # for ES dataset
        train_list = ['806289BC-AD52-4CC1-806C-0CDB14D65EB6',
                     '24E40C4C-A349-4F9F-93AB-01D00FB994AF',
                     '0A986513-7828-4D53-AA1F-E02D6DF9561B',
                     '9DC38D04-E82E-4F29-AB52-B476535226F2',
                     '2C32C23E-E30C-498A-8DD2-0EFB9150A02E',
                     '1155FF54-63D3-4AB2-9863-8385D0BD0A13',
                     '61976C24-1C50-4355-9C49-AAE44A7D09F6',
                     '27E04243-B138-4F40-A164-F40B60165CF3',
                     'ECECC2AB-D32F-4F90-B74C-E12A1C69BBE2',
                     '59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2',
                     '11B5EC4D-4133-4289-B475-4E737182A406',
                     '61359772-D8D8-480D-B623-7C636EAD0C81',
                     '8023FE1A-D3B0-4E2C-A57A-9321B7FC755F',
                     'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C',
                     '5119D0F8-FCA8-4184-A4EB-19421A40DE0D',
                     'A7599A50-24AE-46A6-8EA6-2576F1011D81',
                     '96A358A0-FFF2-4239-B93E-C7425B901B47',
                     'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B',
                     'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A',
                     '74B86067-5D4B-43CF-82CF-341B76BEA0F4',
                     '136562B6-95B2-483D-88DC-065F28409FD2',
                     '59818CD2-24D7-4D32-B133-24C2FE3801E5',
                     '665514DE-49DC-421F-8DCB-145D0B2609AD',
                     'B9724848-C7E2-45F4-9B3F-A1F38D864495',
                     '797D145F-3858-4A7F-A7C2-A4EB721E133C',
                     '481F4DD2-7689-43B9-A2AA-C8772227162B',
                     'E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3',
                     '99B204C0-DD5C-4BB7-83E8-A37281B8D769',
                     'B09E373F-8A54-44C8-895B-0039390B859F',
                     'CCAF77F0-FABB-4F2F-9E24-D56AD0C5A82F',
                     'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96',
                     '1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842',
                     'FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF',
                     '83CF687B-7CEC-434B-9FE8-00C3D5799BE6',
                     '0BFC35E2-4817-4865-BFA7-764742302A2D',
                     'BEF6C611-50DA-4971-A040-87FB979F3FC1',
                     '40E170A7-607B-4578-AF04-F021C3B0384A',
                     '9759096F-1119-4E19-A0AD-6F16989C7E1C',
                     '4E98F91F-4654-42EF-B908-A3389443F2E7',
                     '4FC32141-E888-4BFF-8804-12559A491D8C',
                     'CDA3BBF7-6631-45E8-85BA-EEB416B32A3C',
                     '1538C99F-BA1E-4EFB-A949-6C7C47701B20',
                     '3600D531-0C55-44A7-AE95-A7A38519464E',
                     '5EF64122-B513-46AE-BCF1-E62AAC285D2C',
                     '86A4F379-B305-473D-9D83-FC7D800180EF',
                     '33A85C34-CFE4-4732-9E73-0A7AC861B27A',
                     '5152A2DF-FAF3-4BA8-9CA9-E66B32671A53',
                     '7D9BB102-A612-4E2A-8E22-3159752F55D8',
                     ]

        valid_list = [ '00EABED2-271D-49D8-B599-1D4A09240601',
                     'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC',
                     'BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC',
                     '0E6184E1-90C0-48EE-B25A-F1ECB7B9714E',
                     '81536B0A-8DBF-4D8A-AC24-9543E2E4C8E0',
                     'C48CE857-A0DD-4DDB-BEA5-3A25449B2153',
                     '78A91A4E-4A51-4065-BDA7-94755F0BB3BB',
                     'F50235E0-DD67-4F2A-B00B-1F31ADA998B9',
                     'A5A30F76-581E-4757-97A2-957553A2C6AA',
                     '098A72A5-E3E5-4F54-A152-BBDA0DF7B694',
                     '7CE37510-56D0-4120-A1CF-0E23351428D2',
                     'CA820D43-E5E2-42EF-9798-BE56F776370B'
                    ]
    
#     ============  split train/valid based on no overlapping context:  ============ 
    train_ind = dataset_split(pd_data, train_list)
    valid_ind = dataset_split(pd_data, valid_list)
    
    
    ### splitting method 2: total random
    # ============ random split train/valid:  ============ 
#     random_ind = np.random.rand(pd_data.shape[0])
#     train_ind = random_ind>=0.8
#     valid_ind = ~train_ind
    
    # ==================================================== 
    
#     ### splitting method 3: based on q_struct
#     uniq_struct = ( pd_data.question_structure.unique() )
#     print('Total unique Q structure num: ',  len(uniq_struct))
#     # split the unique Q-struct to 50%-50%
#     rd_num = np.random.rand(len(uniq_struct))
#     train_ind_struct = rd_num<0.8
#     test_ind_struct = rd_num>=0.8

#     train_qstruct = uniq_struct[train_ind_struct]  
#     # valid_qstruct = uniq_struct[valid_ind]  
#     test_qstruct = uniq_struct[test_ind_struct]  
#     train_ind = pd_data.question_structure.isin(train_qstruct)
#     valid_ind = pd_data.question_structure.isin(test_qstruct)

#     print('Train/test split:  %d / %d' %(sum(train_ind), sum(valid_ind)) )
#     # ==================================================== 


    
    sensory_matrix_train =  sensory_matrix[train_ind]
    question_matrix_train =  question_matrix[train_ind]
    answer_matrix_train = answer_matrix[train_ind]

    sensory_matrix_val =  sensory_matrix[valid_ind]
    question_matrix_val =  question_matrix[valid_ind]
    answer_matrix_val = answer_matrix[valid_ind]

    # sensory_matrix_test =  sensory_matrix[test_ind]
    # question_matrix_test =  question_matrix[test_ind]
    # answer_matrix_test = answer_matrix[test_ind]

    print(sensory_matrix_train.shape, question_matrix_train.shape, answer_matrix_train.shape)
    print('\n')
    print(sensory_matrix_val.shape, question_matrix_val.shape, answer_matrix_val.shape)
    print('\n')
    # print(sensory_matrix_test.shape, question_matrix_test.shape, answer_matrix_test.shape)
    # print('\n')

    # saving to file
    processed_split_path = data_folder +  save_name + '_split.npz'
    print('Saving processed data to: ', processed_split_path)

    np.savez(processed_split_path, 
             s_train = sensory_matrix_train, q_train =question_matrix_train, a_train =answer_matrix_train,
             s_val = sensory_matrix_val, q_val =question_matrix_val, a_val =answer_matrix_val,
    #          s_test = sensory_matrix_test, q_test =question_matrix_test, a_test =answer_matrix_test,
             train_ind = train_ind,  valid_ind = valid_ind
            )
    print('Data saved at: ', processed_split_path)

    # npzfile = np.load(processed_test_data_path)
    # print(npzfile.files) 
    return