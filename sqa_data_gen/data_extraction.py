import csv
import numpy as np
import gzip;
from io import StringIO, BytesIO


################## extract opp data to npy from .dat file #############################
def read_opp_files(datapath, filelist, cols, label2id):
    data = []
    labels = []
    for i, filename in enumerate(filelist):
        nancnt = 0
        print('reading file %d of %d' % (i+1, len(filelist)))
#         with open(datapath.rstrip('/') + '/dataset/%s' % filename, 'r') as f:
        with open(datapath +'/%s' % filename, 'r') as f:

            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                elem = []
                for ind in cols:
                    elem.append(line[ind])
                # we can skip lines that contain NaNs, as they occur in blocks at the start
                # and end of the recordings.
                if sum([x == 'NaN' for x in elem]) == 0:
                    data.append([float(x) / 1000 for x in elem[:-1]])
                    labels.append(label2id[elem[-1]])
    return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)+1}



################## extract extra sensory data #############################
def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index(b'\n')];
    columns = headline.split(b',');

    # The first column should be timestamp:
    assert columns[0] == b'timestamp';
    # The last column should be label_source:
    assert columns[-1] == b'label_source';
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith(b'label:'):
            first_label_ind = ci;
            break;
        pass;

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind];
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1];
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith(b'label:');
        label_names[li] = label.replace(b'label:',b'');
        pass;
    
    return (feature_names,label_names);

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(BytesIO(csv_str),delimiter=',',skiprows=1);
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int);
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)];
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1]; # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat); # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0.; # Y is the label matrix
    
    return (X,Y,M,timestamps);

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(uuid, data_path):
    user_data_file = '%s.features_labels.csv.gz' % uuid;

    file_path = data_path + user_data_file
    
    # Read the entire csv file of the user:
    with gzip.open(file_path,'rb') as fid:
        csv_str = fid.read();
        pass;

    (feature_names,label_names) = parse_header_of_csv(csv_str);
    n_features = len(feature_names);
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features);
    
    # standardize X data:
    data_stats_path = 'trained_models/es_model/data_stats.npz'
    npzfile = np.load(data_stats_path)
    data_min=npzfile['data_min']
    denominator=npzfile['denominator']
    X = (X - data_min )/ denominator
    
    return (X,Y,M,timestamps,feature_names,label_names);





################## other helper functions #############################
def visualize_data_labels(label_y, label_list, show_other = True):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    width = 16
    height = 4
    fig= plt.figure(figsize=(width, height))
    
    # if ES dataset:
    if len(label_list) != 3:
        show_other = show_other
        label_list_mid = label_list[0]
        label_mid= label_y[0]
        
#         ax = fig.add_subplot(gs[0])
        tick_marks = np.arange(len(label_list_mid))+1
        plt.yticks(tick_marks, label_list_mid)
        plt.plot(label_mid, '.')
        if not show_other:
            plt.ylim([1.5, 8])
        plt.grid()
        
        return
    
    # if opp dataset:
    show_other = show_other
    label_list_mid, label_list_loc, label_list_high = label_list
    label_mid, label_loc, label_high= label_y
    
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 0.8, 0.8]) 

    ax = fig.add_subplot(gs[0])
    tick_marks = np.arange(len(label_list_mid))+1
    plt.yticks(tick_marks, label_list_mid)
    plt.plot(label_mid, '.')
    if not show_other:
        plt.ylim([1.5, 19])
    plt.grid()

    ax = fig.add_subplot(gs[1])
    tick_marks = np.arange(len(label_list_loc))+1
    plt.yticks(tick_marks, label_list_loc)
    plt.plot(label_loc, '.')
    if not show_other:
        plt.ylim([1.5, 6])
    plt.grid()

    ax = fig.add_subplot(gs[2])
    tick_marks = np.arange(len(label_list_high))+1
    plt.yticks(tick_marks, label_list_high)
    plt.plot(label_high, '.')
    if not show_other:
        plt.ylim([1.5, 7])    
    plt.grid()
    
    return
    

    
    
################## main data extraction function, supporting opportunity and Extrasensory dataset #############################
def extract_data_from_file(filename, datapath, plot_option = True, show_other = False, source_data = 'opp'):
    """
    For differnet source dataset, extract different data: opp or es
    
    Visualize the lablels temporal distribution of opportunity dataset.
    Input:
    filename of the opportunity data file.
    Output:
    Data labels (numerical values for all time steps) and label list for all 3 lvl labels(mid, locomotive, high)
    also return: extracted IMU data
    """
    
    # if source data is ExtraSensory:
    if source_data == 'es':
        
        (X,Y,_,_,_,_) = read_user_data(filename, datapath)
          
        # SWD: data preprocessing:
        # use only mutually exclusive labels 
        label_ind = [0, 1, 2, 3, 4, 44]
        new_Y = np.zeros([Y.shape[0], 7])
        new_Y[:, 1:7] = Y[:, label_ind]
        new_Y[:, 0] = ( Y[:, label_ind].sum(axis = 1) != 1) # put other to the 1st class
        label_y = new_Y.argmax(axis = 1)

        # X features also need preprocessing, but not here. (X is not used in QA generation)
        data_x = X

        label_list = ['do other activities', 'lie down', 'sit down','walk','run','ride bicycle','stand up']
        label_list = [label_list, label_list]  # duplicate lvl1 content as lvl2, in accordance with opp data format

        data_y = [label_y+1, label_y+1] # duplicate lvl1 content as lvl2, in accordance with opp data format
        
        # visualize data labels of a single levels
        if plot_option:
            visualize_data_labels(data_y, label_list, show_other = show_other)
        
        return data_y, label_list, data_x
    
    
    # Otherwise use source data as Opportunity:
#     datapath = 'dataset'
    filelist = [filename]

    #### mid lvl actions ###
    ########################
    mid_label_map = [
        (0,      'Other'),
        (406516, 'Open Door 1'),
        (406517, 'Open Door 2'),
        (404516, 'Close Door 1'),
        (404517, 'Close Door 2'),
        (406520, 'Open Fridge'),
        (404520, 'Close Fridge'),
        (406505, 'Open Dishwasher'),
        (404505, 'Close Dishwasher'),
        (406519, 'Open Drawer 1'),
        (404519, 'Close Drawer 1'),
        (406511, 'Open Drawer 2'),
        (404511, 'Close Drawer 2'),
        (406508, 'Open Drawer 3'),
        (404508, 'Close Drawer 3'),
        (408512, 'Clean Table'),
        (407521, 'Drink from Cup'),
        (405506, 'Toggle Switch')
    ]
    mid_label2id = {str(x[0]): i for i, x in enumerate(mid_label_map)}
    mid_id2label = [x[1] for x in mid_label_map]
    mid_cols = [
        38, 39, 40, 41, 42, 43, 44, 45, 46, # InertialMeasurementUnit BACK
        51, 52, 53, 54, 55, 56, 57, 58, 59, # InertialMeasurementUnit RUA 
        64, 65, 66, 67, 68, 69, 70, 71, 72, # InertialMeasurementUnit RLA 
        77, 78, 79, 80, 81, 82, 83, 84, 85, # InertialMeasurementUnit LUA
        90, 91, 92, 93, 94, 95, 96, 97, 98, # InertialMeasurementUnit LLA
        103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116,  # IMU L+R shoe
        117, 118, 119, 120, 121, 122, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 
        250]
    # lablels of different granularities: 244-locomotion, 245-HL act, 250-ML act
    mid_cols = [x-1 for x in mid_cols] # labels for 18 activities (including other)

    data = read_opp_files(datapath, filelist, mid_cols, mid_label2id)
    label_mid = data['targets']
    label_list_mid = [i[1] for i in mid_label_map]

    #### loc lvl actions ###
    ########################
    loc_label_map = [
        (0, 'Other'),
        (1, 'Stand'),
        (2, 'Walk'),
        (4, 'Sit'),
        (5, 'Lie')
    ]
    loc_label2id = {str(x[0]): i for i, x in enumerate(loc_label_map)}
    loc_id2label = [x[1] for x in loc_label_map]
    loc_cols = [
        38, 39, 40, 41, 42, 43, 44, 45, 46, # InertialMeasurementUnit BACK
        51, 52, 53, 54, 55, 56, 57, 58, 59, # InertialMeasurementUnit RUA 
        64, 65, 66, 67, 68, 69, 70, 71, 72, # InertialMeasurementUnit RLA 
        77, 78, 79, 80, 81, 82, 83, 84, 85, # InertialMeasurementUnit LUA
        90, 91, 92, 93, 94, 95, 96, 97, 98, # InertialMeasurementUnit LLA
        103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116,  # IMU L+R shoe
        117, 118, 119, 120, 121, 122, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 
        244]
    # lablels of different granularities: 244-locomotion, 245-HL act, 250-ML act
    loc_cols = [x-1 for x in loc_cols] # labels for 18 activities (including other)

    data = read_opp_files(datapath, filelist, loc_cols, loc_label2id)
    label_loc = data['targets']
    label_list_loc = [i[1] for i in loc_label_map]

    #### high lvl actions ###
    ########################
    high_label_map = [
        (0,      'Other'),
        (101, 'Relaxing'),
        (102, 'Coffee time'),
        (103, 'Early morning'),
        (104, 'Cleanup'),
        (105, 'Sandwich time')
    ]
    high_label2id = {str(x[0]): i for i, x in enumerate(high_label_map)}
    high_id2label = [x[1] for x in high_label_map]
    high_cols = [
        38, 39, 40, 41, 42, 43, 44, 45, 46, # InertialMeasurementUnit BACK
        51, 52, 53, 54, 55, 56, 57, 58, 59, # InertialMeasurementUnit RUA 
        64, 65, 66, 67, 68, 69, 70, 71, 72, # InertialMeasurementUnit RLA 
        77, 78, 79, 80, 81, 82, 83, 84, 85, # InertialMeasurementUnit LUA
        90, 91, 92, 93, 94, 95, 96, 97, 98, # InertialMeasurementUnit LLA
        103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116,  # IMU L+R shoe
        117, 118, 119, 120, 121, 122, 123, 124,
        125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 
        245]
    # lablels of different granularities: 244-locomotion, 245-HL act, 250-ML act
    high_cols = [x-1 for x in high_cols] # labels for 18 activities (including other)

    data = read_opp_files(datapath, filelist, high_cols, high_label2id)
    label_high = data['targets']
    data_x = data['inputs']
    label_list_high = [i[1] for i in high_label_map]
    
    data_y = [label_mid, label_loc, label_high]
    label_list = [label_list_mid, label_list_loc, label_list_high]
    
    # visualize data labels of different levels
    if plot_option:
        visualize_data_labels(data_y, label_list, show_other = show_other)
    
    return data_y, label_list, data_x



################### split data into pre-defined segements ############################

def data_split_tools(data_x, label_y, length, startpoint = None):
    
    if startpoint is not None:
        if startpoint+length > data_x.shape[0]:
            raise ValueError("SWD: startpoint +  length > data_x.shape!!!")
        seg_data_x = data_x[startpoint:startpoint+length]
        seg_label_y = [i[startpoint:startpoint+length] for i in label_y]
        return seg_data_x, seg_label_y, startpoint
        
    from random import randint
    startpoint = randint(0, data_x.shape[0]-length)
    seg_data_x = data_x[startpoint:startpoint+length]
    seg_label_y = [i[startpoint:startpoint+length] for i in label_y]
    
    return seg_data_x, seg_label_y, startpoint



################### get scene lists for splitted segements ############################

def draw_scene_graph(state_list, state_duration, start_time, label_list):
    
    for time, duration, state in zip(start_time, state_duration, state_list):
        print('['+ str(round(time,1)) +']'+str(label_list[state-1])+' '+str(round(duration,2)) , end='')
        print('==>', end='\n')
#         print('==>', end='')
#     totaltime = len(state_list)/30
    totaltime = 60
    print('END ['+ str(round(totaltime,1)) +']')
    print('\n')
    return

def series2graph(label_series, label_list, show_graph = True, show_other = False):
    """
    input:
    label_series: a list of labels
    label_list: the mapping between labels and encodings.
    show_graph = True: print the generated graph.
    
    Output:
    state_list: index for each state. (Need to have index for union and intersection operations)
    state_list: extracted states from label series
    state_duration: the duration for each state in state_list
    start_time: the start time of each state.
    
    """
    sample_freq = 30
    
    state_list = []
    state_duration = []
    start_time = []
    prev_state = None
    
    for idx, state in enumerate(label_series):
        if prev_state!= state:
            state_list.append(state)
            start_time.append(idx/sample_freq)
            prev_state = state
            
    state_duration = start_time.copy()
    state_duration.pop(0)
    state_duration.append(len(label_series)/sample_freq)
    state_duration = np.array(state_duration) - np.array(start_time)
    
    
    if not show_other:
        other_list = []
        for idx, state in enumerate(state_list):
            if state == 1:
                other_list.append(idx)

        state_list = [state_list[j] for j in range(len(state_list)) if j not in other_list]
        start_time = [start_time[j] for j in range(len(start_time)) if j not in other_list]
        state_duration = [state_duration[j] for j in range(len(state_duration)) if j not in other_list]
        
        
    if show_graph:
        draw_scene_graph(state_list, state_duration, start_time, label_list) 
            
#     return state_list, state_duration, start_time   
    state_index = np.arange(len(state_list))
    return np.array([state_index, state_list, state_duration, start_time])   # state_list, state_duration, start_time     



################### extract sensory scene data from using question info ############################


