import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils import data

from torchsummary import summary
from tqdm import tqdm

import numpy as np
import time
import sys
import os
import pickle
from collections import Counter

from sqa_models.mac_model.dataset import CLEVR, collate_data, transform
from sqa_models.mac_model.model import MACNetwork



class SQA_data(data.Dataset):
    def __init__(self,  data_path, split='train', transform=None):
        
        processed_test_data_path = data_path
        npzfile = np.load(processed_test_data_path)
#         print(npzfile.files)
        self.data_s_split = npzfile['s_' + split]
        self.data_a_split = npzfile['a_' + split]
        self.data_q_split = npzfile['q_' + split]
        
        # adjust dimension
        self.data_a_split = self.data_a_split.argmax(1)
        self.data_s_split = np.expand_dims(self.data_s_split, -1)  
        self.data_s_split = np.swapaxes(self.data_s_split,1,2)
#         self.data_s_split = np.expand_dims(self.data_s_split, -1)

        self.split = split  # train or val

    def __getitem__(self, index):
        
        data_s = self.data_s_split[index]
        data_q = self.data_q_split[index]
        data_a = self.data_a_split[index]

        return data_s, data_q, len(data_q), data_a
    
    def __len__(self):
        return len(self.data_a_split)
    
    
    
def accumulate(model1, model2, decay=0.99):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch, training_set, batch_size, 
          net_running, net, 
          criterion, optimizer, 
          device):
#     training_set = My_Data2(split='train')  # loadding data, time consuming: 2mins
    train_set = DataLoader(
        training_set, batch_size=batch_size, num_workers=1, shuffle = True
#         , collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0
    
    net.train(True)
    
    for iter_id, (image, question, q_len, answer) in enumerate(pbar):
        
        image = image.type(torch.FloatTensor) # change data type: double to float
        q_len = q_len.tolist()
        question = question.type(torch.LongTensor)
        
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size
        
        # correct is the acc for current batch, moving_loss is the acc for previous batches
        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = (moving_loss * iter_id + correct)/(iter_id+1)
#             moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            'Epoch: {}; Curr_Loss: {:.5f}; Curr_Acc: {:.5f}; Tot_Acc(running): {:.5f}'.format(
                epoch + 1, loss.item(), correct, moving_loss
            )
        )
        accumulate(net_running, net)


def valid(training_set, batch_size, 
          net_running, 
          criterion,
          device):
    
    valid_set = DataLoader(
        training_set, batch_size=batch_size, num_workers=1
    )
    
    dataset = iter(valid_set)

#     net_running.train(False)
    family_correct = Counter()
    family_total = Counter()
    loss_total = 0
    output_label = []
    
    with torch.no_grad():
        for image, question, q_len, answer in tqdm(dataset):
            
            family = [1]*len(image)
            image = image.type(torch.FloatTensor) # change data type: double to float
            q_len = q_len.tolist()
            question = question.type(torch.LongTensor)
            
            image, question = image.to(device), question.to(device)

            net_running.eval()
            output = net_running(image, question, q_len)
            loss = criterion(output, answer.to(device))
            loss_total = loss_total + loss
            
            correct = output.detach().argmax(1) == answer.to(device)
            output_label.append(output.detach().argmax(1).cpu().numpy())  # getting output of validation set
            
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    avg_acc = sum(family_correct.values()) / sum(family_total.values())
    avg_loss = (loss_total / sum(family_total.values())).cpu().numpy()
    
    print(
        'Avg Acc: {:.5f}; Avg Loss: {:.5f}'.format(
            avg_acc,
            avg_loss
        )
    )
    output_label = np.concatenate(output_label)

    print('%d / %d'%(sum(family_correct.values()), sum(family_total.values())))
    return avg_acc, avg_loss, output_label # getting output of validation set





def run_mac_model(dataset_path, 
                  hyper_parameters,
                  epochs,
                  model_save_folder,
                  result_save_name,
                  source_data = 'opp'
                  ):
    
    #loading dataset
    print('Loading dataset: ')
    since = time.time()
#     data_path = 'sqa_data/test_split.npz'
    train_set = SQA_data(data_path = dataset_path, 
                         split='train')
    val_set = SQA_data(data_path = dataset_path,
                       split='val')
    print('Dataset loaded using %.2f seconds!\n'%(time.time()-since))
    
    # building network
    n_words = hyper_parameters['n_words'] #400001
    dim = hyper_parameters['dim'] #512
    glove_embeding = hyper_parameters['glove_embeding'] #False
    ebd_train = hyper_parameters['ebd_train'] #True
    n_answers = hyper_parameters['n_answers'] #27 # where to modify: no where...? Q_len, no need to modify. !!!!!!! change this one!
    dropout = hyper_parameters['dropout'] #0.15

    batch_size = hyper_parameters['batch_size'] #64
    learning_rate = hyper_parameters['learning_rate'] #1e-4
    weight_decay = hyper_parameters['weight_decay'] #1e-4

    n_epoch = epochs # 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = MACNetwork(n_words, dim, 
                     vocabulary_embd = glove_embeding, embd_train = ebd_train,
                     classes = n_answers, dropout=dropout, source_data = source_data).to(device)
    net_running = MACNetwork(n_words, dim, 
                             vocabulary_embd = glove_embeding, embd_train = ebd_train,
                             classes = n_answers, dropout=dropout, source_data = source_data).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)


    # saving path:
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    model_name = 'mac_model'
    save_model_name = model_save_folder + model_name + '.model'
    print('Saving model to: ', model_save_folder )
    print('Saving learning result to: ', result_save_name )


    # Begin training!
    
    # import warnings
        # warnings.filterwarnings('ignore')
        
    print('============ Training details: ============ ')
    print('---- Model structure ----')
    print('Hidden dim: ', dim, 'Output dim: ', n_answers )
    print('GLOVE embedding:', glove_embeding,  '   Trainable:', ebd_train)
    print('---- Training details ----')
    print('Dropout:', dropout, '   Weight regularization lambda:', weight_decay )
    print('Batch_size:', batch_size, '   Learning_rate:', learning_rate,  '   Epochs:', n_epoch )
    print('\n\n')

    acc_best = 0.0
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    

    for epoch in range(n_epoch):
        print('==========%d epoch =============='%(epoch+1))
        train(epoch, train_set, batch_size, net_running, net, criterion, optimizer, device)
        
        print('----- Training Acc: ----- ')
        train_acc, train_loss, _ = valid(train_set, batch_size, net_running, criterion, device)
        print('----- Validation Acc: ----- ')
        val_acc, val_loss, _ = valid(val_set, batch_size, net_running, criterion, device)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        if val_acc > acc_best:
            with open(
                save_model_name, 'wb'
    #             'checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb'
            ) as f:
                torch.save(net_running.state_dict(), f) ### should save net_running instead of net!
            print('!!!! Best accuracy increased from %.4f to %.4f !  Saved to: %s.' %(acc_best, val_acc, save_model_name))
            acc_best = val_acc
        else:
            print('Acc not increasing...')

    print('The best validation accuracy: ', acc_best)
    train_hist = np.array([train_acc_list, train_loss_list, val_acc_list, val_loss_list]).T
    
    
    # getting inference result
    print('\n\n\n========================================')
    print('Loading best model from: ',save_model_name)
    checkpoint = torch.load(save_model_name)
    net_running.load_state_dict(checkpoint)
    net_running.eval()
    
    
    _, _, out_label = valid(val_set, batch_size, net_running, criterion, device)
    
    
    # convert out_label to out_correctness!
    npzfile = np.load(dataset_path)
    y_true = npzfile['a_val'].argmax(axis = 1)
    result_correctness = y_true == out_label
    print('Checking: the testing acc is ', result_correctness.sum()/result_correctness.shape[0])
    
    
    save_result = { 'history': train_hist,
                    'inference_result': result_correctness.astype(int)
                    }
    
    with open(result_save_name, 'wb') as handle:
        pickle.dump(save_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Inference result saved to:', result_save_name)
    
    return