import numpy as np
import time
import os
import pickle
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import preprocess_data.embedding as ebd
from sqa_models.baselines import *


def run_baselines(processed_data_path, 
                  pickle_data_path,
                  dl_model_save_folder,
                  epochs_num,
                  result_name,
                  train_dl = True,
                  source_data = 'opp'
                     ):
    
    #     Baseline performance:
    # 0. Neural Symbolic approach (perfect logics)
    # 1. Prior and Prior Q
    # 2 Question only (Blind LSTM)
    # 3. Context only (Deaf CNN)
    # 4. ConvLSTM (concatenate)
    # 5. ConvLSTM (multiply)
    # 6. Stacked Attention Net
    
    
    # loading processed data:
    since = time.time()
    npzfile = np.load(processed_data_path)
    print(npzfile.files)
    
    data_s_train = npzfile['s_train']
    data_a_train = npzfile['a_train']
    data_q_train = npzfile['q_train']

    data_s_valid = npzfile['s_val']
    data_a_valid = npzfile['a_val']
    data_q_valid = npzfile['q_val']
    # data_s_test = npzfile['s_test']
    # data_a_test = npzfile['a_test']
    # data_q_test = npzfile['q_test']
    
    # adjust dimension
    data_s_train = np.expand_dims(data_s_train, -1)
    data_s_valid = np.expand_dims(data_s_valid, -1)
    # data_s_test = np.expand_dims(data_s_test, -1)
    data_s_train = np.swapaxes(data_s_train,1,2)
    data_s_valid = np.swapaxes(data_s_valid,1,2)
    # data_s_test = np.swapaxes(data_s_test,1,2)
    
    print('data loaded:')
    print('Training:')
    print('Sensory matrix: ', data_s_train.shape)
    print('Question matrix: ', data_q_train.shape)
    print('Answer matrix: ', data_a_train.shape)
    print('\nValidation:')
    print('Sensory matrix: ', data_s_valid.shape)
    print('Question matrix: ', data_q_valid.shape)
    print('Answer matrix: ', data_a_valid.shape)
    # print('\nTesting:')
    # print('Sensory matrix: ', data_s_test.shape)
    # print('Question matrix: ', data_q_test.shape)
    # print('Answer matrix: ', data_a_test.shape)
    print('Loading processed data took: %d seconds' %(time.time()-since) )

    
    # loading Pickle QA data
    load_pd_data = pd.read_pickle(pickle_data_path)
    print('\nPickle data loaded at: ',pickle_data_path)
    print('Data shape: ', load_pd_data.shape)
    
    # load embedding matrix
    embedding_matrix = ebd.load()
    print('\nLoading embd matrix shape: ', embedding_matrix.shape)
    
    # =========================== start evaluation ============================
    print('\n=================================')
    print('====== Baseline Evaluation =======')
    print('=================================\n')
    # index for tessting data
    train_ind = npzfile['train_ind']
    valid_ind = npzfile['valid_ind']
    # test_ind = npzfile['test_ind']
    
    ## oracle method:
    result_test_oracle_ans = (load_pd_data[valid_ind].pred_answer==load_pd_data[valid_ind].answer)
    print('\nBaseline Accuracy: \n Oracle: %.4f '\
          %(np.sum(result_test_oracle_ans)/result_test_oracle_ans.shape[0]*100)) 


    # Prior and Prior Q method:
    dict_prior_q, dict_prior = baseline_prior_q(pickle_data_path, train_ind) # get stats using training data
    prior_ans, prior_q_ans = run_baseline_prior(pickle_data_path, 
                                                dict_prior_q, 
                                                dict_prior,
                                                data_ind = valid_ind)
    print('\nBaseline Accuracy: \n Prior: %.4f \n Prior Q: %.4f '\
          %(np.sum(prior_ans)/prior_ans.shape[0]*100, 
            np.sum(prior_q_ans)/prior_q_ans.shape[0]*100))


    result_test_oracle_ans = np.array(result_test_oracle_ans)
    result_test_prior_ans = np.array(prior_ans)
    result_test_prior_q_ans = np.array(prior_q_ans)


    # DL based method:
    dl_model_save_path = 'trained_models/' + dl_model_save_folder
    if not os.path.exists(dl_model_save_path):
        os.makedirs(dl_model_save_path)
        print('Creating dir: ', dl_model_save_path)
            
    if train_dl:
        print('\n=================================================:')
        print('============ Training DL baselines ================: ')
        print('===================================================\n')
        num_words = 400001
        embedding_dim = 300

        num_hidden_lstm = 128
        output_dim =128
        dropout_rate = 0.5
        
        
#         if source_data == 'opp':
#             sen_dim = 77
#             sen_win_len = 1800 
#         else:  # es data
#             sen_dim = 225
#             sen_win_len = 200 # 200 
            
        sen_dim, sen_win_len = data_s_valid.shape[1], data_s_valid.shape[2]  
        
        sen_channel = 1
        num_feat_map = 64

        seq_length = data_q_valid.shape[1] 
        num_classes = data_a_valid.shape[1]
        
        # all baseline methods
        baselines = ['lstm', 'cnn_lstm_cat', 'san', 'cnn_lstm_mul',   'cnn']  
        # Training time: san 555s, cnn542s, lstm 650s, convlstm_mul 640s, cnnlstm_cat: 566s
        epochs_num = epochs_num
        batch_size= 64
        
        train_hist = dict()
        for baseline_type_i in baselines:

            print('\n====================================')
            print('Running baseline: ', baseline_type_i)

            model_hist = train_baseline(embedding_matrix, num_words, embedding_dim,  seq_length,
                                        num_hidden_lstm, output_dim, dropout_rate,
                                        sen_dim, sen_win_len, sen_channel,num_feat_map, 
                                        num_classes,
                                        baseline_type_i,
                                        data_s_train, data_q_train, data_a_train, # training 
                                        data_s_valid, data_q_valid, data_a_valid, # validation

                                        model_save_folder = dl_model_save_path,
                                        epochs = epochs_num,
                                        batch_size=batch_size,
                                        visualize = False
                                       )
            train_hist[baseline_type_i] = model_hist
            
        print('\nDL model training complete!')
        print('=================================\n')
        
    
    # get all inference result and save to 'result/'
    print('\n=================================')
    print('==========Getting all inference===========')
    print('=================================\n')
    dl_model_ressut = get_model_result(dl_model_save_path, 
                              data_s_valid, data_q_valid, data_a_valid,   # inferencing on valid dataset
                              test_batch_size = 64 )
    
    test_data_num = result_test_prior_ans.shape[0]
    test_result = np.zeros([8, test_data_num])

    test_result[0,:] = result_test_prior_ans
    test_result[1,:] = result_test_prior_q_ans
    test_result[2,:] = result_test_oracle_ans
    test_result[3:8,:] = dl_model_ressut

    
    # test_result contains: Prior, Prior_Q, Oracle,  'cnn', 'lstm', 'cnn_lstm_mul', 'cnn_lstm_cat', 'san'
    print('\n=================================')
    print('The shape of result: ', test_result.shape)

    ## saving test result:
    sim_result = dict()
    sim_result['train_curve'] = train_hist
    sim_result['index'] = {'train':train_ind, 'valid':valid_ind }
    sim_result['prior_dict'] = {'prior':dict_prior, 'prior_q':dict_prior_q }
    sim_result['inference'] = test_result
    

    result_path = 'result/' + result_name +'.pkl'
#     np.savez(result_path, test_result)
#     sim_result.to_pickle(result_path)
    
    with open(result_path, 'wb') as handle:
        pickle.dump(sim_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('saving result to: ', result_path)
    
    return 




def train_baseline(embedding_matrix, num_words, embedding_dim,  seq_length,
                    num_hidden_lstm, output_dim, dropout_rate,
                    sen_dim, sen_win_len, sen_channel,num_feat_map, 
                    num_classes,
                    baseline_type,
                    data_s_train, data_q_train, data_a_train,  # train data
                    data_s_val, data_q_val, data_a_val,        # valid data
                    model_save_folder,
                    epochs = 1,
                    batch_size=64,
                    visualize = False
                   ):
    
    model_name = baseline_type
    
    if baseline_type == 'cnn_lstm_mul':
        model = create_baseline_sqa(embedding_matrix, num_words, embedding_dim,  seq_length,
                                        num_hidden_lstm, output_dim, dropout_rate,
                                        sen_dim, sen_win_len, sen_channel,num_feat_map, 
                                        num_classes,
                                        model_type = 'cnn_lstm_mul'
                                       )
            
        train_x = [data_s_train, data_q_train]
        train_y = data_a_train
        valid_x = [data_s_val, data_q_val]
        valid_y = data_a_val
        
    if baseline_type == 'cnn_lstm_cat':
        model = create_baseline_sqa(embedding_matrix, num_words, embedding_dim,  seq_length,
                                        num_hidden_lstm, output_dim, dropout_rate,
                                        sen_dim, sen_win_len, sen_channel,num_feat_map, 
                                        num_classes,
                                        model_type = 'cnn_lstm_cat'
                                       )
            
        train_x = [data_s_train, data_q_train]
        train_y = data_a_train
        valid_x = [data_s_val, data_q_val]
        valid_y = data_a_val
        
    if baseline_type == 'lstm':
        model = create_baseline_sqa(embedding_matrix, num_words, embedding_dim,  seq_length,
                                        num_hidden_lstm, output_dim, dropout_rate,
                                        sen_dim, sen_win_len, sen_channel,num_feat_map, 
                                        num_classes,
                                        model_type = 'lstm'
                                       )
            
        train_x = [data_q_train]
        train_y = data_a_train
        valid_x = [data_q_val]
        valid_y = data_a_val
        
    if baseline_type == 'cnn':
        model = create_baseline_sqa(embedding_matrix, num_words, embedding_dim,  seq_length,
                                        num_hidden_lstm, output_dim, dropout_rate,
                                        sen_dim, sen_win_len, sen_channel,num_feat_map, 
                                        num_classes,
                                        model_type = 'cnn'
                                       )
            
        train_x = [data_s_train]
        train_y = data_a_train
        valid_x = [data_s_val]
        valid_y = data_a_val
        
    if baseline_type == 'san':
        model = create_SAN_model(embedding_matrix, num_words, embedding_dim,  seq_length,
                                num_hidden_lstm, output_dim, dropout_rate,
                                sen_dim, sen_win_len, sen_channel,num_feat_map, 
                                num_classes,
                                num_glimpses=2, n=1
                                )           
        train_x = [data_s_train, data_q_train]
        train_y = data_a_train
        valid_x = [data_s_val, data_q_val]
        valid_y = data_a_val

        
    if visualize:
        model.summary()

        
    ############## training ##############   
    save_path = model_save_folder+ model_name+'.hdf5'
    # es = EarlyStopping(monitor='val_MAE', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(save_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    cb_list = [mc]
    epochs = epochs
    batch_size= batch_size

    model.compile(optimizer='adam', 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    print('============ Start Opp Model Training ===========\n')
    print('The maximum training epochs is: ', epochs)

    H = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        callbacks = cb_list,
#                         validation_split = 0.2)
                        validation_data=(valid_x, valid_y ) )
    
    K.clear_session()
    del model
                  
    return H.history


def get_testing_ans(model_name, save_path,
                    data_s, data_q, data_a,
                    batch_size = 96
                   ):
    
    if model_name == 'cnn' or model_name == 'cnn0':
        train_x = [data_s]
        train_y = data_a
    elif model_name == 'lstm':
        train_x = [ data_q]
        train_y = data_a
    else:
        train_x = [data_s, data_q]
        train_y = data_a
    
    trained_model_1 = load_model(save_path)
    print('Model loaded: ', model_name)

    # evaluate saved model
    y_pred1 = trained_model_1.predict(train_x, batch_size=batch_size, verbose=1)
    y_pred1= np.argmax(y_pred1, axis=1)
    y_true1 = np.argmax(train_y, axis=1)

    y_ans = (y_pred1==y_true1)
    
    K.clear_session()
    del trained_model_1
    
    return y_ans


def get_model_result(model_save_folder, data_s, data_q, data_a, test_batch_size = 96 ):
    """
    """
    
    test_data_num = data_a.shape[0]
    test_result = np.zeros([5, test_data_num])
    
    model_list = ['cnn', 'lstm', 'cnn_lstm_mul', 'cnn_lstm_cat', 'san']   # 5 DL models
    
    for idx, model_i in enumerate(model_list):
        
        save_path_i = model_save_folder + model_i+'.hdf5'

        ans_i = get_testing_ans(model_i, save_path_i,
                           data_s, data_q, data_a,
                          test_batch_size
                           )
        print('Model %s accuracy: %.3f%%' %(model_i ,ans_i.sum()/ans_i.shape[0]*100) )
        test_result[idx, :] = ans_i
        
    return test_result