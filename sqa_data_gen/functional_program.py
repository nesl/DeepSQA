from sqa_data_gen.function_catalog import *

# question families as a dictionary
function_families = {} 

# 'Did the user <A>?'
def function_0(action_1, scene, valid_ext = False):
    
    node_1 = fc_filter_action_type(action_1, scene)
    node_2 = fc_exist(node_1)
    
    return node_2
function_families[0] = function_0


# 'How many times did the user <A>?',
def function_1(action_1, scene, valid_ext = False):
    
    node_1 = fc_filter_action_type(action_1, scene)
    node_2 = fc_count_num(node_1)
    
    return node_2
function_families[1] = function_1


# 'Did the user <A> [relation] <B>?'
def function_2(action_1, action_2, relation, scene, question_validation, valid_ext = False):
    
    node_1 = fc_relate(scene, action_2, relation, valid_ext)
    node_2 = fc_filter_action_type(action_1, node_1)
    node_3 = fc_exist(node_2)
    
    if question_validation:
        intermediate_node = [node_1[0]]
        if any(elem is None for elem in intermediate_node):
            raise ValueError("SWD: Question Degenerated!!!") 
    
    return node_3

function_families[2] = function_2



# Did the user <A> [relation] <B> [logic] [relation] <C>?
def function_3(action_1, action_2, action_3, relation_1, relation_2, logic, scene, question_validation, valid_ext = False):
    
    node_1 = fc_relate(scene, action_2, relation_1, valid_ext)
    node_2 = fc_relate(scene, action_3, relation_2, valid_ext )
    if logic == 'AND':
        node_3 = fc_and(node_1, node_2)
    elif logic == 'OR':
        node_3 = fc_or(node_1, node_2)
    else:
        raise ValueError("SWD: Invalid Logic!")
        
    node_4 = fc_filter_action_type(action_1, node_3)  
    node_5 = fc_exist(node_4)
    
    if question_validation:
        intermediate_node = [node_1[0], node_2[0], node_3[0]]
        if any(elem is None for elem in intermediate_node):
            raise ValueError("SWD: Question Degenerated!!!") 
            
    return node_5

function_families[3] = function_3



# How many times did the user <A> [relation] <B>?'
def function_4(action_1, action_2, relation, scene, question_validation, valid_ext = False):
    
    node_1 = fc_relate(scene, action_2, relation, valid_ext)
    node_2 = fc_filter_action_type(action_1, node_1)
    node_3 = fc_count_num(node_2)
    
    if question_validation:
        intermediate_node = [node_1[0]]
        if any(elem is None for elem in intermediate_node):
            raise ValueError("SWD: Question Degenerated!!!") 
    
    return node_3

function_families[4] = function_4



# How many times did the user <A> [relation] <B> [logic] [relation] <C> ?'
def function_5(action_1, action_2, action_3, relation_1, relation_2, logic, scene, question_validation, valid_ext = False):
    
    node_1 = fc_relate(scene, action_2, relation_1, valid_ext )
    node_2 = fc_relate(scene, action_3, relation_2, valid_ext )
    if logic == 'AND':
        node_3 = fc_and(node_1, node_2)
    elif logic == 'OR':
        node_3 = fc_or(node_1, node_2)
    else:
        raise ValueError("SWD: Invalid Logic!")
        
    node_4 = fc_filter_action_type(action_1, node_3)  
    node_5 = fc_count_num(node_4)
    
    if question_validation:
        intermediate_node = [node_1[0], node_2[0], node_3[0]]
        if any(elem is None for elem in intermediate_node):
            raise ValueError("SWD: Question Degenerated!!!") 

    return node_5

function_families[5] = function_5



# 'What did the user do [relation] <A>?'  # for right_before, or right_after
def function_6(action_1, relation, scene, valid_ext = False):
    
    node_1 = fc_relate(scene, action_1, relation, valid_ext)
    node_2 = fc_unique(node_1, valid_ext)
    node_3 = fc_query_action_type(node_2)
    
    return node_3

function_families[6] = function_6



# 'What did the user do [relation1] <A> and [relation2] <B>?'
def function_7(action_1, action_2, relation_1, relation_2, scene, question_validation, valid_ext = False):
    
    node_1 = fc_relate(scene, action_1, relation_1, valid_ext)
    node_2 = fc_relate(scene, action_2, relation_2, valid_ext)
    node_3 = fc_and(node_1, node_2)
    node_4 = fc_unique(node_3, valid_ext)
    node_5 = fc_query_action_type(node_4)
    
    if question_validation:
        intermediate_node = [node_1[0], node_2[0]]
        if any(elem is None for elem in intermediate_node):
            raise ValueError("SWD: Question Degenerated!!!") 
    
    return node_5

function_families[7] = function_7



# 'Did the user  perform the same(different) action right before and after <A>?'
def function_8(action_1, scene, valid_ext = False):
    
    node_1 = fc_relate(scene, action_1, 'Preceding', valid_ext)
    node_2 = fc_relate(scene, action_1, 'Following', valid_ext)
    node_3 = fc_unique(node_1, valid_ext)
    node_4 = fc_unique(node_2, valid_ext)
    node_5 = fc_query_action_type(node_3)
    node_6 = fc_query_action_type(node_4)
    node_7 = fc_action_equal(node_5, node_6)
    
    return node_7

function_families[8] = function_8



# 'Did the user  perform the same/different action [relation1] <A> and [relation2] <B>?'
def function_9(action_1, action_2, relation_1, relation_2, scene, valid_ext = False):
    
    node_1 = fc_relate(scene, action_1, relation_1, valid_ext)
    node_2 = fc_relate(scene, action_2, relation_2, valid_ext)
    node_3 = fc_unique(node_1, valid_ext)
    node_4 = fc_unique(node_2, valid_ext)
    node_5 = fc_query_action_type(node_3)
    node_6 = fc_query_action_type(node_4)
    node_7 = fc_action_equal(node_5, node_6)
    
    return node_7

function_families[9] = function_9



# Did the user  <A>  for the same(different) times before and after <B>?'
def function_10(action_1, action_2, scene, valid_ext = False):
    
    node_1 = fc_relate(scene, action_2, 'Before', valid_ext)
    node_2 = fc_relate(scene, action_2, 'After', valid_ext)
    node_3 = fc_filter_action_type(action_1, node_1)
    node_4 = fc_filter_action_type(action_1, node_2)
    node_5 = fc_count_num(node_3)
    node_6 = fc_count_num(node_4)
    node_7 = fc_equal(node_5, node_6)
    
    return node_7

function_families[10] = function_10



# Did the user open the door 1 more than open the door 2?'
def function_11(action_1, action_2, scene, valid_ext = False):
    
    node_1 = fc_filter_action_type(action_1, scene)
    node_2 = fc_filter_action_type(action_2, scene)
    node_3 = fc_count_num(node_1)
    node_4 = fc_count_num(node_2)
    node_5 = fc_more(node_3, node_4)
    
    return node_5

function_families[11] = function_11



# What did the user do While he <A1>?',
def function_12(action_1, scene, valid_ext = False):
    
    node_1 = fc_relate(scene, action_1, 'While', valid_ext)
    node_2 = fc_unique(node_1, valid_ext)
    node_3 = fc_query_action_type(node_2)
    
    return node_3

function_families[12] = function_12



# How many times did the user <A1> While he <A2>?',
def function_13(action_1, action_2, scene, question_validation, valid_ext = False):
    
    node_1 = fc_relate(scene, action_2, 'While', valid_ext)
    node_2 = fc_filter_action_type(action_1, node_1)
    node_3 = fc_count_num(node_2)
    
    if question_validation:
        intermediate_node = [node_1[0]]
        if any(elem is None for elem in intermediate_node):
            raise ValueError("SWD: Question Degenerated!!!") 
    
    return node_3

function_families[13] = function_13



# How long did the user <A1>?',
def function_14(action_1, scene, valid_ext = False):
    
    node_1 = fc_filter_action_type(action_1, scene)
    node_2 = fc_count_time(node_1)
    
    return node_2

function_families[14] = function_14



# How long did the user <A1> [R1] <A2>?',
def function_15(action_1, action_2, relation_1, scene, question_validation, valid_ext = False):
    
    node_1 = fc_relate(scene, action_2, relation_1, valid_ext)
    node_2 = fc_filter_action_type(action_1, node_1)
    node_3 = fc_count_time(node_2)
    
    if question_validation:
        intermediate_node = [node_1[0]]
        if any(elem is None for elem in intermediate_node):
            raise ValueError("SWD: Question Degenerated!!!") 
    
    return node_3

function_families[15] = function_15



