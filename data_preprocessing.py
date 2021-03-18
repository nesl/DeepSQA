from preprocess_data.preprocessing import *

data_name = 's1234_1500_400_balanced.pkl'
context_data = 's1234_1500_400_context.pkl'
save_name = 'opp_sim8'

print('Source data: ', data_name)
print('context data: ', context_data)
print('Processed data name: ', save_name)

preprocess_data(data_name, 
            save_name,
            data_folder = 'sqa_data/', 
            create_ebd = False,
            glove_path = 'glove/glove.6B.300d.txt',
            context_name = context_data,
            source_data = 'opp',
            win_len = 1500
                )