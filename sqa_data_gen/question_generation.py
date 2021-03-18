import numpy as np
import random
import json

from pattern import en

from sqa_data_gen.function_catalog import *
from sqa_data_gen.functional_program import *

# Alternative approach: specify the tense (https://www.clips.uantwerpen.be/pages/pattern-en)
def fixSentenceTense(question,actions,relations,combinators):
    
    text = question[0]
    actionCount = 1
    
    #Replace the actions
    for action in actions:
        # verb should be first word of action
        # swd: if there's only one word, then handle differently
        action_split = action.split(' ')
        verb = action_split[0] 
        # get tense:
        verb = en.conjugate(verb, tense=question[actionCount][0], person=question[actionCount][1], number=question[actionCount][2])
        replacedAction = '<A'+str(actionCount)+'>'
        if len( action_split )== 1:
            text = text.replace(
                text[text.find(replacedAction):(text.find(replacedAction)+len(replacedAction))], 
                                                   verb)
        else:
            text = text.replace(
                text[text.find(replacedAction):(text.find(replacedAction)+len(replacedAction))], 
                                                   verb + action[action.find(' '):])
        actionCount = actionCount + 1
    
    #Replace the relations
    relationCount = 1
    for relation in relations:
        replacedRelation = '[R'+str(relationCount)+']'
#         print("Before replacement ("+replacedRelation+"): ", text)
        text = text.replace(text[text.find(replacedRelation):(text.find(replacedRelation)+len(replacedRelation))], 
                                               relation)
#         print("After replacement: ",text)
        relationCount = relationCount + 1
    
    #Replace the combinators
    combinatorCount = 1
    for combinator in combinators:
        replacedCombinator = '[C'+str(combinatorCount)+']'
        text = text.replace(text[text.find(replacedCombinator):(text.find(replacedCombinator)+len(replacedCombinator))], 
                                               combinator)
        combinatorCount = combinatorCount + 1
        
    return text


def question_generator(scene_lists, scene_lists_pred,
                       question_family_file, 
                       label_list,
                       show_other = False,
                       question_validation = True,
                       source_data = 'opp',
                       diagnose = False 
                      ):
    
    """
    Generate questions and answers in NLP form.
    
    Input: 
    scene_lists: the semantic representation of sensory context data. (2 lvls)
    scene_lists_pred: the predicted semantic representation of sensory context.
    question_family_file: the filename of question family (JSON format)
    label_list: the corresponding label_list for the annotations in scene_lists.
    
    Output:
    
    
    """
    
    def question_summary(q_counter, q_all_num, q_id):
        print('===================================')
        print('Generate %d/%d questions in total of question family %d.'%(q_counter, q_all_num, q_id))
        return
    
    # loading question family from JSON
    with open(question_family_file) as json_file:
        question_family = json.load(json_file)
    
    # diagnose print
    if diagnose:
        print('===================================')
        print('Starting generating questions: ')
        print('Validation : ', question_validation)
        print('Size of question families: ', len(question_family['questions'] ))
        
    # dict for change binary answer to natural language
    answer_dict = {'True': 'Yes', 'False': 'No'}
    
    # all possible relations and logics
    relation_family = ['Before', 'After', 'Preceding', 'Following']
    logic_combinator_family = ['AND', 'OR']
    
    # all involvded actions and locomotions  # changed to all possible actions
    # whether using "other" as one of the activity: yes... change 2 places
    if source_data == 'opp':
        if show_other:
            unique_actions = np.array( range(1, 19) ) # in total 18 classes, 1-18
            unique_loc = np.array( range(1,6) )
        else:
            unique_actions = np.array( range(2, 19) )  # the 1st class is other
            unique_loc = np.array( range(2,6) )
    elif source_data == 'es':
        if show_other:
            unique_actions = np.array( range(1, 8) )  # in total 7 classes, 1-7
            unique_loc = np.array( range(1,2) )
        else:
            unique_actions = np.array( range(2, 8) )  # the 1st class is other
            unique_loc = np.array( range(1,2) )
    else:
        print('== Wrong source data type! ==')
        return
        
#     unique_actions = np.unique( scene_lists[0][1] )
#     unique_loc = np.unique( scene_lists[1][1] )



    # diagnose print
    if diagnose:
        print('\n===================================')
        print('Unique actions: ')
        print( unique_actions)
        print('Unique locomotions: ')
        print( unique_loc)
        
    
    # initialize question/ans/q_tpye lists:
    question_family_index = []
    question_nl = []
    answer_nl = []
    answer_nl_p = [] # store the predicted answer from Neural-symbolic appraoch
    question_struct = []  # store the structure of question. In the form of a string: ID_act1_act2_combine_relate1_relate2
    
    # generating questions: if using an automatic approach, need to use recursion. 
    
    # =========================== question 0 ===========================
    q_id = 0
    q_counter = 0
    q_all_num = len(unique_actions)**1 * len(relation_family)**0 * len(logic_combinator_family)**0 

    for action_1 in unique_actions:
        actions = [label_list[0][int(action_1)-1]]
        
        
        q_f_nlp = question_family['questions'][q_id]['texts']
        question_id = random.randint(0,len(q_f_nlp)-1)
        question_nlp = q_f_nlp[question_id]
        
        question_nlp = fixSentenceTense(question_nlp, actions,[],[])
            #question_nlp.replace('<A1>', action_1_nlp)

        # Try generating questions for all possible combinations
        try:         
            ans_sm = str( function_families[q_id](action_1 , scene_lists) )
            ans_nlp = answer_dict[ans_sm]

    #         print(question_nlp, ans_nlp)
            question_family_index.append(q_id)
            question_nl.append(question_nlp)
            answer_nl.append(ans_nlp)
            question_struct.append(str(q_id)+'_'+str(actions))
            
            # Try generating answers for the question
            try:
                ans_sm_p = str( function_families[q_id](action_1 , scene_lists_pred, valid_ext = True) )
                ans_nlp_p = answer_dict[ans_sm_p]
            except ValueError:
                ans_nlp_p = 'Invalid'
            answer_nl_p.append(ans_nlp_p)

            q_counter += 1
        except ValueError:  ####### The question is not valid in the context
            pass
        

    if diagnose:
        question_summary(q_counter, q_all_num, q_id)


    # =========================== question 1 ===========================
    q_id = 1
    q_counter = 0
    q_all_num = len(unique_actions)**1 * len(relation_family)**0 * len(logic_combinator_family)**0 

    for action_1 in unique_actions:
        actions = [label_list[0][int(action_1)-1]]

        q_f_nlp = question_family['questions'][q_id]['texts']
        question_id = random.randint(0,len(q_f_nlp)-1)
        question_nlp = q_f_nlp[question_id]
        #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
        question_nlp = fixSentenceTense(question_nlp, actions,[],[])
        
        # Try generating questions for all possible combinations
        try:         
            ans_sm = str( function_families[q_id](action_1 , scene_lists) )
            ans_nlp = ans_sm

            #         print(question_nlp, ans_nlp)
            question_family_index.append(q_id)
            question_nl.append(question_nlp)
            answer_nl.append(ans_nlp)
            question_struct.append(str(q_id)+'_'+str(actions))
            
            # Try generating answers for the question
            try:
                ans_sm_p = str( function_families[q_id](action_1 , scene_lists_pred, valid_ext = True) )
                ans_nlp_p = ans_sm_p
            except ValueError:
                ans_nlp_p = 'Invalid'
            answer_nl_p.append(ans_nlp_p)

            q_counter += 1
        except ValueError:  ####### The question is not valid in the context
            pass  

    if diagnose:
        question_summary(q_counter, q_all_num, q_id)


    # =========================== question 2 ===========================
    # for question 2
    q_id = 2
    q_counter = 0
    q_all_num = len(unique_actions)**2 * len(relation_family)**1 * len(logic_combinator_family)**0 

    for relation in relation_family:
        for action_1 in unique_actions:
            for action_2 in unique_actions:
                if action_2 == action_1: # avoid meaningless questions
                    continue

                actions = [label_list[0][int(action_1)-1],label_list[0][int(action_2)-1]]
                #action_2_nlp = label_list[0][int(action_2)-1]

                q_f_nlp = question_family['questions'][q_id]['texts']
                question_id = random.randint(0,len(q_f_nlp)-1)
                question_nlp = q_f_nlp[question_id]
                question_nlp = fixSentenceTense(question_nlp, actions,[relation],[])
                #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                #question_nlp = question_nlp.replace('<A2>', action_2_nlp)
                #question_nlp = question_nlp.replace('[R1]', relation)

                # Try generating questions for all possible combinations
                try:         
                    ans_sm = str( function_families[q_id](action_1, action_2, 
                                                          relation, 
                                                          scene_lists,
                                                          question_validation) )
                    ans_nlp = answer_dict[ans_sm]

            #         print(question_nlp, ans_nlp)
                    question_family_index.append(q_id)
                    question_nl.append(question_nlp)
                    answer_nl.append(ans_nlp)
                    question_struct.append(str(q_id)+'_'+str(actions) + '_' + relation)
                    
                    # Try generating answers for the question
                    try:
                        ans_sm_p = str( function_families[q_id](action_1, action_2, 
                                                          relation, 
                                                          scene_lists_pred,
                                                          question_validation = False, valid_ext = True) )
                        ans_nlp_p = answer_dict[ans_sm_p]
                    except ValueError:
                        ans_nlp_p = 'Invalid'
                    answer_nl_p.append(ans_nlp_p)

                    q_counter += 1
                except ValueError:  ####### The question is not valid in the context
                    pass

    if diagnose:
        question_summary(q_counter, q_all_num, q_id)

    # =========================== question 3 ===========================
    # for question 3
    q_id = 3
    q_counter = 0
    q_all_num = len(unique_actions)**3 * len(relation_family)**2 * len(logic_combinator_family)**1 

    for c_logic_1 in logic_combinator_family:
        for relation_1 in relation_family:
            for relation_2 in relation_family:
                for action_1 in unique_actions:
                    for action_2 in unique_actions:
                        if action_2 == action_1: # avoid meaningless questions
                            continue
                        for action_3 in unique_actions:
                            if action_3 == action_1: # avoid meaningless questions
                                continue
                            if action_2 == action_3 and relation_1 == relation_2: # avoid meaningless questions
                                continue
                            

                            actions =  [label_list[0][int(action_1)-1], label_list[0][int(action_2)-1], label_list[0][int(action_3)-1]]
                            relations = [relation_1, relation_2]
                            combinators = [c_logic_1]
                            #action_1_nlp = label_list[0][int(action_1)-1]
                            #action_2_nlp = label_list[0][int(action_2)-1]
                            #action_3_nlp = label_list[0][int(action_3)-1]

                            q_f_nlp = question_family['questions'][q_id]['texts']
                            question_id = random.randint(0,len(q_f_nlp)-1)
                            question_nlp = q_f_nlp[question_id]
                            question_nlp = fixSentenceTense(question_nlp, actions, relations, combinators)
                            #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                            #question_nlp = question_nlp.replace('<A2>', action_2_nlp)
                            #question_nlp = question_nlp.replace('<A3>', action_3_nlp)
                            #question_nlp = question_nlp.replace('[R1]', relation_1)
                            #question_nlp = question_nlp.replace('[R2]', relation_2)
                            #question_nlp = question_nlp.replace('[C1]', c_logic_1)

                            # Try generating questions for all possible combinations
                            try:
                                ans_sm = function_families[q_id](action_1, action_2, action_3, 
                                                                 relation_1, relation_2, 
                                                                 c_logic_1, 
                                                                 scene_lists,
                                                                 question_validation) 
                                ans_sm = str( ans_sm )
                                ans_nlp = answer_dict[ans_sm]

                        #         print(question_nlp, ans_nlp)
                                question_family_index.append(q_id)
                                question_nl.append(question_nlp)
                                answer_nl.append(ans_nlp)
                                question_struct.append(str(q_id)+'_'+str(actions) + '_' + str(relations)+ '_'+str(combinators) )

                                # Try generating answers for the question
                                try:
                                    ans_sm_p = function_families[q_id](action_1, action_2, action_3, 
                                                                 relation_1, relation_2, 
                                                                 c_logic_1, 
                                                                 scene_lists_pred,
                                                                 question_validation = False, valid_ext = True)
                                    ans_nlp_p = answer_dict[str(ans_sm_p)]
                                except ValueError:
                                    ans_nlp_p = 'Invalid'
                                answer_nl_p.append(ans_nlp_p)
                                
                                q_counter += 1
                            except ValueError:  ####### The question is not valid in the context
                                pass   


    if diagnose:
        question_summary(q_counter, q_all_num, q_id)



    # =========================== question 4 ===========================
    # for question 4
    q_id = 4
    q_counter = 0
    q_all_num = len(unique_actions)**2 * len(relation_family)**1 * len(logic_combinator_family)**0 

    for relation_1 in relation_family:
        for action_1 in unique_actions:
            for action_2 in unique_actions:
                if action_1 == action_2: # avoid meaningless questions
                    continue

                actions = [label_list[0][int(action_1)-1],label_list[0][int(action_2)-1]]
                relations = [relation_1]
                #action_1_nlp = label_list[0][int(action_1)-1]
                #action_2_nlp = label_list[0][int(action_2)-1]

                q_f_nlp = question_family['questions'][q_id]['texts']
                question_id = random.randint(0,len(q_f_nlp)-1)
                question_nlp = q_f_nlp[question_id]

                question_nlp = fixSentenceTense(question_nlp, actions, relations, [])
                #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                #question_nlp = question_nlp.replace('<A2>', action_2_nlp)
                #question_nlp = question_nlp.replace('[R1]', relation_1)

                # Try generating questions for all possible combinations
                try:
                    ans_sm = function_families[q_id](action_1, action_2, 
                                                     relation_1, 
                                                     scene_lists,
                                                     question_validation) 
                    ans_sm = str( ans_sm )
    #                 ans_nlp = answer_dict[ans_sm]
                    ans_nlp = ans_sm

            #         print(question_nlp, ans_nlp)
                    question_family_index.append(q_id)
                    question_nl.append(question_nlp)
                    answer_nl.append(ans_nlp)
                    question_struct.append(str(q_id)+'_'+str(actions) + '_' + str(relations) )
                    
                    # Try generating answers for the question
                    try:
                        ans_sm_p = function_families[q_id](action_1, action_2, 
                                                     relation_1, 
                                                     scene_lists_pred,
                                                     question_validation = False, valid_ext = True)
                        ans_nlp_p = str(ans_sm_p)
                    except ValueError:
                        ans_nlp_p = 'Invalid'
                    answer_nl_p.append(ans_nlp_p)
                                

                    q_counter += 1
                except ValueError:  ####### The question is not valid in the context
                    pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id)




    # =========================== question 5 ===========================
    # for question 5
    q_id = 5
    q_counter = 0
    q_all_num = len(unique_actions)**3 * len(relation_family)**2 * len(logic_combinator_family)**1 

    for c_logic_1 in logic_combinator_family:
        for relation_1 in relation_family:
            for relation_2 in relation_family:
                for action_1 in unique_actions:
                    for action_2 in unique_actions:
                        if action_1 == action_2: # avoid meaningless questions
                            continue
                        for action_3 in unique_actions:
                            if action_1 == action_3: # avoid meaningless questions
                                continue
                            if action_2 == action_3 and relation_1 == relation_2: # avoid meaningless questions
                                continue
                            

                            actions = [label_list[0][int(action_1)-1], label_list[0][int(action_2)-1], label_list[0][int(action_3)-1]]
                            relations = [relation_1, relation_2]
                            combinators = [c_logic_1]
                            #action_1_nlp = label_list[0][int(action_1)-1]
                            #action_2_nlp = label_list[0][int(action_2)-1]
                            #action_3_nlp = label_list[0][int(action_3)-1]

                            q_f_nlp = question_family['questions'][q_id]['texts']
                            question_id = random.randint(0,len(q_f_nlp)-1)
                            question_nlp = q_f_nlp[question_id]
                            
                            question_nlp = fixSentenceTense(question_nlp, actions, relations, combinators) 
                            #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                            #question_nlp = question_nlp.replace('<A2>', action_2_nlp)
                            #question_nlp = question_nlp.replace('<A3>', action_3_nlp)
                            #question_nlp = question_nlp.replace('[R1]', relation_1)
                            #question_nlp = question_nlp.replace('[R2]', relation_2)
                            #question_nlp = question_nlp.replace('[C1]', c_logic_1)

                            # Try generating questions for all possible combinations
                            try:
                                ans_sm = function_families[q_id](action_1, action_2, action_3, 
                                                                 relation_1, relation_2, 
                                                                 c_logic_1, 
                                                                 scene_lists,
                                                                 question_validation) 
                                ans_sm = str( ans_sm )
                #                 ans_nlp = answer_dict[ans_sm]
                                ans_nlp = ans_sm

                                        #         print(question_nlp, ans_nlp)
                                question_family_index.append(q_id)
                                question_nl.append(question_nlp)
                                answer_nl.append(ans_nlp)
                                question_struct.append(str(q_id)+'_'+str(actions) + '_' + str(relations)+ '_'+str(combinators) )

                                # Try generating answers for the question
                                try:
                                    ans_sm_p = function_families[q_id](action_1, action_2, action_3, 
                                                                 relation_1, relation_2, 
                                                                 c_logic_1, 
                                                                 scene_lists_pred,
                                                                 question_validation = False, valid_ext = True) 
                                    ans_nlp_p = str( ans_sm_p )
                                except ValueError:
                                    ans_nlp_p = 'Invalid'
                                answer_nl_p.append(ans_nlp_p)
                                
                                q_counter += 1
                            except ValueError:  ####### The question is not valid in the context
                                pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id) 




    # =========================== question 6 ===========================
    # for question 6
    q_id = 6
    q_counter = 0
    q_all_num = len(unique_actions)**1 * len(relation_family)**1 * len(logic_combinator_family)**0 

    for relation_1 in relation_family:
        for action_1 in unique_actions:

            #action_1_nlp = label_list[0][int(action_1)-1]
            actions = [label_list[0][int(action_1)-1]]
            relations = [relation_1]

            q_f_nlp = question_family['questions'][q_id]['texts']
            question_id = random.randint(0,len(q_f_nlp)-1)
            question_nlp = q_f_nlp[question_id]

            
            question_nlp = fixSentenceTense(question_nlp, actions, relations, []) 
            #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
            #question_nlp = question_nlp.replace('[R1]', relation_1)

            # Try generating questions for all possible combinations
            try:
                ans_sm = function_families[q_id](action_1,
                                                 relation_1, 
                                                 scene_lists )
    #             ans_sm = str( ans_sm )
        #                 ans_nlp = answer_dict[ans_sm]
                ans_nlp = label_list[0][int(ans_sm)-1]

            #         print(question_nlp, ans_nlp)
                question_family_index.append(q_id)
                question_nl.append(question_nlp)
                answer_nl.append(ans_nlp)
                question_struct.append(str(q_id)+'_'+str(actions) + '_' + str(relations) )
                
                # Try generating answers for the question
                try:
                    ans_sm_p = function_families[q_id](action_1,
                                                 relation_1, 
                                                 scene_lists_pred , valid_ext = True)
                    ans_nlp_p = label_list[0][int(ans_sm_p)-1]
                except ValueError:
                    ans_nlp_p = 'Invalid'
                answer_nl_p.append(ans_nlp_p)

                q_counter += 1
            except ValueError:  ####### The question is not valid in the context
                pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id)


    # =========================== question 7 ===========================
    # for question 7
    q_id = 7
    q_counter = 0
    q_all_num = len(unique_actions)**2 * len(relation_family)**2 * len(logic_combinator_family)**0

    for relation_1 in relation_family:
        for relation_2 in relation_family:
            for action_1 in unique_actions:
                for action_2 in unique_actions:
                    if action_2 == action_1: # avoid meaningless questions
                        continue
                    

                    actions = [label_list[0][int(action_1)-1],label_list[0][int(action_2)-1]] 
                    #action_1_nlp = label_list[0][int(action_1)-1]
                    #action_2_nlp = label_list[0][int(action_2)-1]
                    relations = [relation_1, relation_2]

                    q_f_nlp = question_family['questions'][q_id]['texts']
                    question_id = random.randint(0,len(q_f_nlp)-1)
                    question_nlp = q_f_nlp[question_id]
                    
                    question_nlp = fixSentenceTense(question_nlp, actions, relations, []) 
                    #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                    #question_nlp = question_nlp.replace('<A2>', action_2_nlp)
                    #question_nlp = question_nlp.replace('[R1]', relation_1)
                    #question_nlp = question_nlp.replace('[R2]', relation_2)

                    # Try generating questions for all possible combinations
                    try:
                        ans_sm = function_families[q_id](action_1, action_2, 
                                                         relation_1, relation_2, 
                                                         scene_lists,
                                                         question_validation) 

        #                 ans_nlp = answer_dict[ans_sm]
                        ans_nlp = label_list[0][int(ans_sm)-1]

                #         print(question_nlp, ans_nlp)
                        question_family_index.append(q_id)
                        question_nl.append(question_nlp)
                        answer_nl.append(ans_nlp)
                        question_struct.append(str(q_id)+'_'+str(actions) + '_' + str(relations) )
                        
                        # Try generating answers for the question
                        try:
                            ans_sm_p = function_families[q_id](action_1, action_2, 
                                                         relation_1, relation_2, 
                                                         scene_lists_pred,
                                                         question_validation = False, valid_ext = True) 
                            ans_nlp_p = label_list[0][int(ans_sm_p)-1]
                        except ValueError:
                            ans_nlp_p = 'Invalid'
                        answer_nl_p.append(ans_nlp_p)

                        q_counter += 1
                    except ValueError:  ####### The question is not valid in the context
                        pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id)


    # =========================== question 8 ===========================
    # for question 8
    q_id = 8
    q_counter = 0
    q_all_num = len(unique_actions)**1 * len(relation_family)**0 * len(logic_combinator_family)**0

    for action_1 in unique_actions:
        #action_1_nlp = label_list[0][int(action_1)-1]
        actions = [label_list[0][int(action_1)-1]]
        
        q_f_nlp = question_family['questions'][q_id]['texts']
        question_id = random.randint(0,len(q_f_nlp)-1)
        question_nlp = q_f_nlp[question_id]

        question_nlp = fixSentenceTense(question_nlp, actions, [], []) 
        #question_nlp = question_nlp.replace('<A1>', action_1_nlp)

        # Try generating questions for all possible combinations
        try: 
            ans_sm = function_families[q_id](action_1, 
                                             scene_lists)
            ans_sm = str( ans_sm )
            ans_nlp = answer_dict[ans_sm]

    #         print(question_nlp, ans_nlp)
            question_family_index.append(q_id)
            question_nl.append(question_nlp)
            answer_nl.append(ans_nlp)
            question_struct.append(str(q_id)+'_'+str(actions))
            
            # Try generating answers for the question
            try:
                ans_sm_p = function_families[q_id](action_1, 
                                             scene_lists_pred, valid_ext = True)
                ans_nlp_p = answer_dict[str(ans_sm_p)]
            except ValueError:
                ans_nlp_p = 'Invalid'
            answer_nl_p.append(ans_nlp_p)

            q_counter += 1
        except ValueError:  ####### The question is not valid in the context
            pass   

    if diagnose:
        question_summary(q_counter, q_all_num, q_id)



    # =========================== question 9 ===========================
    # for question 9
    q_id = 9
    q_counter = 0
    q_all_num = len(unique_actions)**2 * len(relation_family)**2 * len(logic_combinator_family)**0

    for relation_1 in relation_family:
        for relation_2 in relation_family:
            for action_1 in unique_actions:
                for action_2 in unique_actions:
                    if action_1 == action_2:
                        continue

                    actions = [label_list[0][int(action_1)-1],label_list[0][int(action_2)-1]]
                    #action_1_nlp = label_list[0][int(action_1)-1]
                    #action_2_nlp = label_list[0][int(action_2)-1]
                    relations = [relation_1, relation_2]

                    q_f_nlp = question_family['questions'][q_id]['texts']
                    question_id = random.randint(0,len(q_f_nlp)-1)
                    question_nlp = q_f_nlp[question_id]
                    question_nlp = fixSentenceTense(question_nlp, actions, relations, []) 

                    #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                    #question_nlp = question_nlp.replace('<A2>', action_2_nlp)
                    #question_nlp = question_nlp.replace('[R1]', relation_1)
                    #question_nlp = question_nlp.replace('[R2]', relation_2)

                    # Try generating questions for all possible combinations
                    try:
                        ans_sm = function_families[q_id](action_1, action_2, 
                                                         relation_1, relation_2, 
                                                         scene_lists)

                        ans_sm = str( ans_sm )
                        ans_nlp = answer_dict[ans_sm]

                #         print(question_nlp, ans_nlp)
                        question_family_index.append(q_id)
                        question_nl.append(question_nlp)
                        answer_nl.append(ans_nlp)
                        question_struct.append(str(q_id)+'_'+str(actions) + '_' + str(relations) )
                        
                        # Try generating answers for the question
                        try:
                            ans_sm_p = function_families[q_id](action_1, action_2, 
                                                         relation_1, relation_2, 
                                                         scene_lists_pred, valid_ext = True)
                            ans_nlp_p = answer_dict[str(ans_sm_p)]
                        except ValueError:
                            ans_nlp_p = 'Invalid'
                        answer_nl_p.append(ans_nlp_p)

                        q_counter += 1
                    except ValueError:  ####### The question is not valid in the context
                        pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id)



    # =========================== question 10 ===========================
    # for question 10
    q_id = 10
    q_counter = 0
    q_all_num = len(unique_actions)**2 * len(relation_family)**0 * len(logic_combinator_family)**0 

    for action_1 in unique_actions:
        for action_2 in unique_actions:
            if action_1 == action_2: # avoid meaningless question
                continue
                
            actions = [label_list[0][int(action_1)-1], label_list[0][int(action_2)-1]]
            #action_1_nlp = label_list[0][int(action_1)-1]
            #action_2_nlp = label_list[0][int(action_2)-1]

            q_f_nlp = question_family['questions'][q_id]['texts']
            question_id = random.randint(0,len(q_f_nlp)-1)
            question_nlp = q_f_nlp[question_id]
            question_nlp = fixSentenceTense(question_nlp, actions, [], []) 

            #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
            #question_nlp = question_nlp.replace('<A2>', action_2_nlp)

            # Try generating questions for all possible combinations
            try:
                ans_sm = function_families[q_id](action_1, action_2, 
                                                 scene_lists)
                ans_sm = str( ans_sm )
                ans_nlp = answer_dict[ans_sm]

        #         print(question_nlp, ans_nlp)
                question_family_index.append(q_id)
                question_nl.append(question_nlp)
                answer_nl.append(ans_nlp)
                question_struct.append(str(q_id)+'_'+str(actions) )
                
                # Try generating answers for the question
                try:
                    ans_sm_p = function_families[q_id](action_1, action_2, 
                                                 scene_lists_pred, valid_ext = True)
                    ans_nlp_p = answer_dict[str(ans_sm_p)]
                except ValueError:
                    ans_nlp_p = 'Invalid'
                answer_nl_p.append(ans_nlp_p)

                q_counter += 1
            except ValueError:  ####### The question is not valid in the context
                pass   


    if diagnose:
        question_summary(q_counter, q_all_num, q_id) 



    # =========================== question 11 ===========================
    # for question 11
    q_id = 11
    q_counter = 0
    q_all_num = len(unique_actions)**2 * len(relation_family)**0 * len(logic_combinator_family)**0 

    for action_1 in unique_actions:
        for action_2 in unique_actions:
            if action_1 == action_2: # avoid meaningless question
                continue
                
            actions = [label_list[0][int(action_1)-1], label_list[0][int(action_2)-1]]
            #action_1_nlp = label_list[0][int(action_1)-1]
            #action_2_nlp = label_list[0][int(action_2)-1]

            q_f_nlp = question_family['questions'][q_id]['texts']
            question_id = random.randint(0,len(q_f_nlp)-1)
            question_nlp = q_f_nlp[question_id]
            question_nlp = fixSentenceTense(question_nlp, actions, [], []) 

            #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
            #question_nlp = question_nlp.replace('<A2>', action_2_nlp)

            # Try generating questions for all possible combinations
            try:
                ans_sm = function_families[q_id](action_1, action_2, 
                                                 scene_lists)
                ans_sm = str( ans_sm )
                ans_nlp = answer_dict[ans_sm]

        #         print(question_nlp, ans_nlp)
                question_family_index.append(q_id)
                question_nl.append(question_nlp)
                answer_nl.append(ans_nlp)
                question_struct.append(str(q_id)+'_'+str(actions) )
                
                # Try generating answers for the question
                try:
                    ans_sm_p = function_families[q_id](action_1, action_2, 
                                                 scene_lists_pred, valid_ext = True)
                    ans_nlp_p = answer_dict[str(ans_sm_p)]
                except ValueError:
                    ans_nlp_p = 'Invalid'
                answer_nl_p.append(ans_nlp_p)

                q_counter += 1
            except ValueError:  ####### The question is not valid in the context
                pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id)
        
        
    if source_data == 'opp':  ## only generate while question for opp data
        # =========================== question 12 ===========================
        # for question 12
        q_id = 12
        q_counter = 0
        q_all_num = len(unique_actions)**0 * len(relation_family)**0 * len(logic_combinator_family)**0  * len(unique_loc)**1

        for action_1 in unique_loc:

            actions = [label_list[1][int(action_1)-1]]

            q_f_nlp = question_family['questions'][q_id]['texts']
            question_id = random.randint(0,len(q_f_nlp)-1)
            question_nlp = q_f_nlp[question_id]

    #         question_nlp = question_nlp.replace('<A1>', action_1_nlp)
            question_nlp = fixSentenceTense(question_nlp, actions, [], []) 

            # Try generating questions for all possible combinations
            try:
                ans_sm = function_families[q_id](action_1, 
                                                 scene_lists)
                ans_nlp = label_list[0][int(ans_sm)-1]

        #         print(question_nlp, ans_nlp)
                question_family_index.append(q_id)
                question_nl.append(question_nlp)
                answer_nl.append(ans_nlp)
                question_struct.append(str(q_id)+'_'+str(actions) )

                # Try generating answers for the question
                try:
                    ans_sm_p = function_families[q_id](action_1, 
                                                 scene_lists_pred, valid_ext = True)
                    ans_nlp_p = label_list[0][int(ans_sm_p)-1]
                except ValueError:
                    ans_nlp_p = 'Invalid'
                answer_nl_p.append(ans_nlp_p)

                q_counter += 1
            except ValueError:  ####### The question is not valid in the context
                pass


        if diagnose:
            question_summary(q_counter, q_all_num, q_id)
        
        
    if source_data == 'opp':  ## only generate while question for opp data
        # =========================== question 13 ===========================
        # for question 13
        q_id = 13
        q_counter = 0
        q_all_num = len(unique_actions)**1 * len(relation_family)**0 * len(logic_combinator_family)**0  * len(unique_loc)**1

        for action_1 in unique_actions:
            for action_2 in unique_loc:

                actions = [label_list[0][int(action_1)-1],label_list[1][int(action_2)-1]]
                #action_1_nlp = label_list[0][int(action_1)-1]
                #action_2_nlp = label_list[1][int(action_2)-1]

                q_f_nlp = question_family['questions'][q_id]['texts']
                question_id = random.randint(0,len(q_f_nlp)-1)
                question_nlp = q_f_nlp[question_id]
                question_nlp = fixSentenceTense(question_nlp, actions, [], []) 

                #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                #question_nlp = question_nlp.replace('<A2>', action_2_nlp)

                # Try generating questions for all possible combinations
                try:
                    ans_sm = function_families[q_id](action_1, action_2, 
                                                     scene_lists,
                                                     question_validation)
                    ans_sm = str( ans_sm )
                    ans_nlp = ans_sm

            #         print(question_nlp, ans_nlp)
                    question_family_index.append(q_id)
                    question_nl.append(question_nlp)
                    answer_nl.append(ans_nlp)
                    question_struct.append(str(q_id)+'_'+str(actions) )

                    # Try generating answers for the question
                    try:
                        ans_sm_p = function_families[q_id](action_1, action_2, 
                                                     scene_lists_pred,
                                                     question_validation = False, valid_ext = True)
                        ans_nlp_p = str(ans_sm_p)
                    except ValueError:
                        ans_nlp_p = 'Invalid'
                    answer_nl_p.append(ans_nlp_p)

                    q_counter += 1
                except ValueError:  ####### The question is not valid in the context
                    pass


        if diagnose:
            question_summary(q_counter, q_all_num, q_id)
        
        

    # =========================== question 14 ===========================
    # for question 14
    q_id = 14
    q_counter = 0
    q_all_num = len(unique_actions)**1 * len(relation_family)**0 * len(logic_combinator_family)**0 

    for action_1 in unique_actions:

        actions = [label_list[0][int(action_1)-1]]

        q_f_nlp = question_family['questions'][q_id]['texts']
        question_id = random.randint(0,len(q_f_nlp)-1)
        question_nlp = q_f_nlp[question_id]

        #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
        question_nlp = fixSentenceTense(question_nlp, actions, [], [])

        # Try generating questions for all possible combinations
        try:
            ans_sm = function_families[q_id](action_1, 
                                             scene_lists)
            ans_sm = str( ans_sm )
            ans_nlp = ans_sm

    #         print(question_nlp, ans_nlp)
            question_family_index.append(q_id)
            question_nl.append(question_nlp)
            answer_nl.append(ans_nlp)
            question_struct.append(str(q_id)+'_'+str(actions))
            
            # Try generating answers for the question
            try:
                ans_sm_p = function_families[q_id](action_1, 
                                             scene_lists_pred, valid_ext = True)
                ans_nlp_p = str(ans_sm_p)
            except ValueError:
                ans_nlp_p = 'Invalid'
            answer_nl_p.append(ans_nlp_p)

            q_counter += 1
        except ValueError:  ####### The question is not valid in the context
            pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id)
        
        
        
    # =========================== question 15 ===========================
    # for question 15
    q_id = 15
    q_counter = 0
    q_all_num = len(unique_actions)**2 * len(relation_family)**1 * len(logic_combinator_family)**0 

    for action_1 in unique_actions:
        for action_2 in unique_actions:
            for relation_1 in relation_family:
                if action_1 == action_2: # avoid meaningless question
                    continue

                #action_1_nlp = label_list[0][int(action_1)-1]
                #action_2_nlp = label_list[0][int(action_2)-1]
                actions = [label_list[0][int(action_1)-1], label_list[0][int(action_2)-1]]
                relations = [relation_1]

                q_f_nlp = question_family['questions'][q_id]['texts']
                question_id = random.randint(0,len(q_f_nlp)-1)
                question_nlp = q_f_nlp[question_id]
                question_nlp = fixSentenceTense(question_nlp, actions, relations, [])

                #question_nlp = question_nlp.replace('<A1>', action_1_nlp)
                #question_nlp = question_nlp.replace('<A2>', action_2_nlp)
                #question_nlp = question_nlp.replace('[R1]', relation_1)

                # Try generating questions for all possible combinations
                try:
                    ans_sm = function_families[q_id](action_1, action_2, 
                                                     relation_1,
                                                     scene_lists,
                                                     question_validation)
                    ans_sm = str( ans_sm )
                    ans_nlp = ans_sm

            #         print(question_nlp, ans_nlp)
                    question_family_index.append(q_id)
                    question_nl.append(question_nlp)
                    answer_nl.append(ans_nlp)
                    question_struct.append(str(q_id)+'_'+str(actions) + '_' + str(relations) )
                    
                    # Try generating answers for the question
                    try:
                        ans_sm_p = function_families[q_id](action_1, action_2, 
                                                     relation_1,
                                                     scene_lists_pred,
                                                     question_validation = False, valid_ext = True)
                        ans_nlp_p = str(ans_sm_p)
                    except ValueError:
                        ans_nlp_p = 'Invalid'
                    answer_nl_p.append(ans_nlp_p)

                    q_counter += 1
                except ValueError:  ####### The question is not valid in the context
                    pass


    if diagnose:
        question_summary(q_counter, q_all_num, q_id) 
        
        
    # ====================== Question Generation Finished ================    
    
    return question_family_index, question_nl, answer_nl, answer_nl_p, question_struct
