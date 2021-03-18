# org function catalog with None in scenelist, in use now.
import numpy as np


    
################### Filtering ###################

def fc_filter_action_type(actionID , scene_lists):
    """
    Filter the specific action in the 1st lvl scene_list
    
    Input: 
    scene_lists: 2 scene_list, of 1st and 2nd lvl, each with a 4xN data array 
    (4 dims: ind, actionID, duration, starting_time)
    actionID: the actionID we are filtering
    
    Output:
    new_scene_lists: 2 scene_list, of 1st lvl (filter other actions except for acitonID) and 2nd lvl
    """
    
    scene_list_1 = scene_lists[0]
    if scene_list_1 is None:
        return [None, scene_lists[1]]
        
    action_ind = scene_list_1[1, :] == actionID
    scene_list_1 = scene_list_1[:, action_ind]
    
    if scene_list_1.size == 0:
        scene_list_1 = None
    
    new_scene_lists = [scene_list_1, scene_lists[1]]
    
    return new_scene_lists

    

def fc_relate(scene_lists, actionID, relation, valid_ext = False):
    """
    Return all Actions in the scene 
    that have the specified temporal relation to the input action.
    5 Relation(temporal): before, after, Preceding, Following, while
        "Before": returns the action elements before the specific action
        "After": returns the action elements after the specific action
        "Preceding": returns the single action element right after the specific action
        "Following": returns the single action element right before the specific action
        "While": returns the action elements during the specific action in lvl-2

    Input:
    scene_lists, actionID(can be both lvl-1 or lvl-2 (only for "While")), relation
    
    Output:
    out_scene_lists: scene_lists that satisfy the required relation
    
    Note: the scene_list must only contain a single action of the given actionID, 
          otherwise raise confusion!
          
    Note: For "While" relation, we only return the list of actions whose starting&ending time fall into the given period.
    
    Note: If the given action is already 1st or last one, "Before" and "After" would just give empty
          action lists; For "Preceding" and "Following", also give empty result.
    """
    
    scene_list_1 = scene_lists[0]
    scene_list_2 = scene_lists[1]
    
    if scene_list_1 is None:
        return [None, scene_lists[1]]
    
    if relation == "While":
        scene_list_1_ = np.concatenate( (scene_list_1, scene_list_1[3:4, :]+scene_list_1[2:3:, :]), axis = 0 )
        loc_ind = np.where(scene_list_2[1,:] == actionID) # here the actionID refers to the locomotionID
        start_t = scene_list_2[3, loc_ind]
        end_t = scene_list_2[2, loc_ind] + start_t
        new_scene_list_2 = scene_list_2[:, loc_ind] 
        
        # the scene_list must only contain a single action of the given actionID
        if valid_ext:
            if new_scene_list_2.shape[0] == 0:
                raise ValueError("SWD: No action found of the given actionID")
            new_scene_list_2 = new_scene_list_2[-1, :]  # Modification of extending valid question to general case (use the last action)
        
        if np.prod(new_scene_list_2.shape)==4:
            pass
        else:
            raise ValueError("SWD: The scene_list must only contain a single action of the given actionID")
        
        act_ind = (scene_list_1_[3,:]>=start_t)&(scene_list_1_[4,:]<=end_t)
        new_scene_list_1 = scene_list_1_[0:4, act_ind[0,:]]
        if new_scene_list_1.size == 0:
            return [None, new_scene_list_2]
        out_scene_lists = [new_scene_list_1, new_scene_list_2]
        return out_scene_lists
        
    # for all these 4 relations, the scene_list lvl2 stay unchanged.
    if relation == "Before":
        act_ind = np.where(scene_list_1[1,:] == actionID)[0]
        
        # the scene_list must only contain a single action of the given actionID
        if valid_ext:
            if act_ind.size == 0:
                raise ValueError("SWD: No action found of the given actionID")
            act_ind = act_ind[-1]  # Modification of extending valid question to general case (use the last action)
            
        if np.prod(act_ind.shape)==1:
            act_ind = act_ind.item()
        else:
            raise ValueError("SWD: The scene_list must only contain a single action of the given actionID")
    
        if act_ind == 0:
            scene_list_1 = None
        else:
            scene_list_1 = scene_list_1[:, 0:act_ind]
        return [scene_list_1, scene_list_2]
        
        
    elif relation == "Preceding":
        act_ind = np.where(scene_list_1[1,:] == actionID)[0]
        # the scene_list must only contain a single action of the given actionID
        if valid_ext:
            if act_ind.size == 0:
                raise ValueError("SWD: No action found of the given actionID")
            act_ind = act_ind[-1]  # Modification of extending valid question to general case (use the last action)
            
        if np.prod(act_ind.shape)==1:
            act_ind = act_ind.item()
        else:
            raise ValueError("SWD: The scene_list must only contain a single action of the given actionID")
        
        if act_ind == 0:
            scene_list_1 = None
        else:
            scene_list_1 = scene_list_1[:, act_ind-1:act_ind]
        return [scene_list_1, scene_list_2]
        
    
    elif relation == "After":
        act_ind = np.where(scene_list_1[1,:] == actionID)[0]
        # the scene_list must only contain a single action of the given actionID
        if valid_ext:
            if act_ind.size == 0:
                raise ValueError("SWD: No action found of the given actionID")
            act_ind = act_ind[-1]  # Modification of extending valid question to general case (use the last action)
            
        if np.prod(act_ind.shape)==1:
            act_ind = act_ind.item()
        else:
            raise ValueError("SWD: The scene_list must only contain a single action of the given actionID")
        
        if act_ind == (scene_list_1.shape[1]-1):
            scene_list_1 = None
        else:
            scene_list_1 = scene_list_1[:, act_ind+1:]
        return [scene_list_1, scene_list_2]

    
    elif relation == "Following":
        act_ind = np.where(scene_list_1[1,:] == actionID)[0]
        # the scene_list must only contain a single action of the given actionID
        if valid_ext:
            if act_ind.size == 0:
                raise ValueError("SWD: No action found of the given actionID")
            act_ind = act_ind[-1]  # Modification of extending valid question to general case (use the last action)
            
        if np.prod(act_ind.shape)==1:
            act_ind = act_ind.item()
        else:
            raise ValueError("SWD: The scene_list must only contain a single action of the given actionID")
        
        if act_ind == (scene_list_1.shape[1]-1):
            scene_list_1 = None
        else:
            scene_list_1 = scene_list_1[:, act_ind+1:act_ind+2]
        return [scene_list_1, scene_list_2]
    
        
    else:
        raise ValueError("SWD: Invalid relation input!")
        

        
################### Query ###################   

def fc_unique(scene_lists, valid_ext = False):
    """
    This function returns the single unique action element of lvl-1 scene_list
    It's invalid if the lvl-1 scene_list contains more than 1 unique action_types
    The result action element is usually fed to "fc_query_action_type" to get the action type.
    """
    scene_list_1 = scene_lists[0]
    if scene_list_1 is None:
        raise ValueError("SWD: The input scene_list must contain only 1 unique action_type")
    
    if valid_ext:
        scene_list_1 = scene_list_1[:-1]  # Modification of extending valid question to general case (use the last action)
        
    # The input scene_list must contain only 1 unique action_type
    if np.abs(scene_list_1[1,:] - scene_list_1[1,0]).sum()==0:
        pass
    else:
        raise ValueError("SWD: The input scene_list must contain only 1 unique action_type")
        
    return scene_list_1[:,0]
    
    

def fc_query_action_type(action):
    """
    This function returns the specified attribute of the input action.
    query_action_type (Action → Action_type) 
    Input is a single action tuple:
    Action (index, action_type, action_duration, starting_time)
    Ouput: action_type
    """
    # The input must be a single action tuple!
    if np.prod(action.shape) == 4:
        pass
    else:
        raise ValueError("SWD: The input must be a single action tuple!")
        
    return action[1]



################### Logical Operations ###################

def fc_and(scene_lists_1, scene_lists_2):
    """
    Returns the intersection of the two input lvl 1 scene_lists.
    """
    scene_list_1 = scene_lists_1[0]
    scene_list_2 = scene_lists_2[0]
    
    if (scene_list_1 is None) or (scene_list_2 is None):
        return [None, scene_lists_1[1]]
    
    xy, x_ind, _ = np.intersect1d(scene_list_1[0,:], scene_list_2[0,:], return_indices=True)
    new_scene_list = scene_list_1[:, x_ind]
    
    if new_scene_list.size == 0:
        new_scene_list = None
    
#     # The scene list of lvl-2 must be the same!
#     if (scene_lists_1[1] == scene_lists_2[1]).all():
#         pass
#     else:
#         raise ValueError("SWD: The scene list of lvl-2 must be the same!")
    # swd... : i don;t know the logic now...
        
    return [new_scene_list, scene_lists_1[1]]



def fc_or(scene_lists_1, scene_lists_2):
    """
    Returns the union of the two input sets
    """
    scene_list_1 = scene_lists_1[0]
    scene_list_2 = scene_lists_2[0]
    
    if (scene_list_1 is None) and (scene_list_2 is None):
        return [None, scene_lists_1[1]]
    elif (scene_list_1 is None):
        return scene_lists_2
    elif (scene_list_2 is None):
        return scene_lists_1
    
    concat_scene_list = np.concatenate((scene_list_1,scene_list_2), axis =1)
    new_scene_list, _ = np.unique(concat_scene_list, axis=1, return_index=True) # returns a sorted array based on the value in 1st row (status index)
    
#     # The scene list of lvl-2 must be the same!
#     if (scene_lists_1[1] == scene_lists_2[1]).all():
#         pass
#     else:
#         raise ValueError("SWD: The scene list of lvl-2 must be the same!") 
    # swd... : i don;t know the logic now...

    return [new_scene_list, scene_lists_1[1]]
      


################### action comparison ###################

def fc_action_equal(actionID_1, actionID_2):
    """
    Functions return yes if their inputs are equal and no if they are not equal. 
    """
    if actionID_1 == actionID_2:
        return True
    else:
        return False
    

    
################### Existence ###################

def fc_exist(scene_lists):
    """
    Checking if any action exists in the 1st lvl scene_list
    Return true / False
    """
    scene_list_1 = scene_lists[0]
    if scene_list_1 is None:
        return False
    count = scene_list_1.shape[1]
    if count == 0:
        return False
    else:
        return True
    

################### Counting ###################

def fc_count_num(scene_lists):
    """
    Returns the size of the input lvl-1 scene_list. 
    """
    
    scene_list_1 = scene_lists[0]
    if scene_list_1 is None:
        return 0
    
    return scene_list_1.shape[1]
    

def fc_count_time(scene_lists):
    """
    Returns the total duration of lvl-1 scene_list. 
    """
    scene_list_1 = scene_lists[0]
    if scene_list_1 is None:
        return 0
    
    return scene_list_1.sum(axis = 1)[2]
    
    
    
################### integer comparison ###################

def fc_equal(num1, num2):
    if num1 == num2:
        return True
    else:
        return False

def fc_less(num1, num2):
    if num1 < num2:
        return True
    else:
        return False
    
def fc_more(num1, num2):
    if num1 > num2:
        return True
    else:
        return False

