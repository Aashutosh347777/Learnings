# Refactoring into functions

import numpy as np

# Global vars, constructor assignment
num_rows:int=0
num_cols:int=0
num_classes:int=0
sys_entro = 0
type_list = []
index_cols_data = []
targ_feats = None
max_depth = 0
min_purity = 0
left_set = None
right_set = None

columns_gone:list = []

tree_formation = {}


targ_col:np.array = None
feat_cols:np.array = None

def decision_tree_classify(dataset:np.array,params:dict):
    """
    Final call for getting the classification of the given data. Initializes the data and parameters for global access

    Parameters:
    dataset: A numpy array or a tabular data which the model gets trained on.
    params: Mathematical values required for building the model

    Returns:
    classification_model : A model to classify incoming values, trained on the params provided at the time of call
    """
    global feat_cols, targ_col,targ_feats,num_classes,total_class_freq,sys_entro
    # print(dataset)
    # print(type(dataset))
    manage_data(dataset,params)
    _find_cols_type(feat_cols)
    sys_entro = _sys_entropy(targ_col)
    targ_feats, total_class_freq =_cal_probs(targ_col,return_metrics=True)[1:]
    num_classes= len(total_class_freq)

    return find_best_split(feat_cols,targ_col)

def manage_data(dataset:np.array,params:dict)-> None:
    """
    Ready the data for calculations.

    Parameters: 
    dataset: A numpy array or tabular data. Sets global variables.
    params: Dictionary object holding how the decision tree should be structured.
    
    Returns
    targ_col: Separated target column.
    feat_cols: Separated feature columns.
    """
    global targ_col,feat_cols,num_rows,num_cols,max_depth,index_cols_data

    # dataset dimesion and allocation
    num_rows,num_cols = dataset.shape
    feat_cols,targ_col = dataset[:,:-1],dataset[:,-1]
    
    index_cols_data = [str(i) for i in range(feat_cols.shape[1])]

    # intializing parameters
    max_depth = params['max_depth']
    
    # for now return None
    return None

def _find_cols_type(arr:np.array) -> list:
    """
    Distinguishes if a column is numerical or categorical

    Parameters:
    arr: The feature columns of the dataset

    Returns:
    type_list: A list that specifies whether a columns is numerical or categorical.
    """
    global type_list
    for col in range(arr.shape[1]):
        # print(type(arr[0,col]))
        # kind_num = isinstance(arr[0,col], (int,np.int64))
        try:
            arr[0,col].astype(float)
            kind_num = True
        except ValueError:
            kind_num = False
            
        type_list.append('numeric'if kind_num else 'categorical')
    
    return type_list

def _cal_probs(arr_obj:np.array,return_metrics:bool=False,is_num_cols=False)->np.array:
    """
    Calculated probability for a catergorically sperateable cols.

    Parameters:
    arr_obj: A numpy array object.

    Returns:
    probs: A numpy array object containing probability for unique elements.
    """
    # May not be dynamic, check
    var_counts = np.unique(arr_obj,return_counts=True)
    uniq_ele,freq_count = var_counts

    if return_metrics:
        return (freq_count/np.sum(freq_count),uniq_ele,freq_count) 

    if is_num_cols:
        return arr_obj/np.sum(arr_obj)

    return freq_count/np.sum(freq_count)

def _cal_entropy(probs:np.array)-> float:
    """
    Calculate the entropy for a given probabilites.

    Parameters:
    probs: Porbability for each class.

    Returns:
    entropy: Entropy/ unecertaininty in the system
    """
    return -sum(probs*np.log2(probs))

def _sys_entropy(targ_col:np.array,is_of=False):
    """
    Calculate system entropy.

    Parmeters:
    targ_column: The targetted column for classification

    Returns:
    sys_entro: Entropy value of the system.
    """
    global tree_formation
    cur_sys_entro = _cal_entropy(_cal_probs(targ_col))
    breakpoint()
    if is_of == True:
        tree_formation['targ_col'] = cur_sys_entro
    return cur_sys_entro


def _cal_weighted_entropy(probs:np.array, entropies:np.array):
    """
    Calculate the weighted entropy of given probability and entropy pairs.

    Parameters:
    probs: Probaibility of each concerned entropy object occuring.
    entropies: Entropy of considered condition.
    """
    return sum(probs * entropies)

def is_col_num_correct(a:str,b:str):
    """
    Find out if the assigned column number is correct or not
    Parameters:
    a: String column index which was picked before.
    b: String column index which was picked later.

    Returns:
    ----
    """
    int_a = int(a)
    int_b = int(b)
    
    if int_a==int_b:
        int_b = int_a + 1

    elif int_a > int_b:
        c = int_a -int_b
    
    elif int_b > int_a:
        c = int_a + int_b
    pass

def _cal_ig_col(feat_cols:np.array,targ_col:np.array,sys_entro:float)-> float:
    """
    Calculate weighted entopy for colummns based on the data type categorical or numerical.

    Parameters:
    feat_cols: Column whoose Infromation gain is to be calculate.
    targ_col: Target column of the set.
    sys_entro: calculated system entropy.

    Returns:
    col_ig: Information gained of the feature column.
    """
    global total_class_freq,num_classes,columns_gone,index_cols_data,index_cols_data
    num_rows,num_cols = feat_cols.shape
    #  =_cal_probs(targ_col,return_metrics=True)[2:]
    # num_classes = len(total_class_freq)
    mapping_splits = {}
    breakpoint()
    for col_num in range(len(index_cols_data)):
        col_dtype = type_list[col_num]
        cur_feat_col = feat_cols[:,col_num]
        # differnetiate the column type.
        if col_dtype == "categorical":
            probs_col,uniq_elements,freq_count=_cal_probs(cur_feat_col,return_metrics=True)
            sum_probs = sum(probs_col)
            highest_ig_split,highest_ig = (),0


            for i in range(len(uniq_elements)):
                # binary splitting for a certian element, probability calculation
                probs_in_contra = [probs_col[i],sum_probs-probs_col[i]]

                cur_ele = uniq_elements[i]

                #masking for filtering only the current element
                mask = cur_feat_col==cur_ele
                feat_col_cur_ele, feat_col_not_cur_ele = cur_feat_col[mask],cur_feat_col[~mask]
                correspondent_vals_cur_ele, correspondent_vals_not_cur_ele = targ_col[mask], targ_col[~mask]

                # calculating the probabilities
                proba_curr_ele,proba_not_cur_ele = _cal_probs(correspondent_vals_cur_ele),_cal_probs(correspondent_vals_not_cur_ele)

                # calculating entropy
                entroyp_curr_ele, entropy_not_cur_ele = _cal_entropy(proba_curr_ele), _cal_entropy(proba_not_cur_ele)

                # ordering entropies            
                entropies = np.array([entroyp_curr_ele,entropy_not_cur_ele])
                curr_ig = sys_entro-_cal_weighted_entropy(probs_in_contra,entropies)
                # print(f"IG {curr_ig}")
                if curr_ig > highest_ig: 
                    highest_ig = curr_ig    
                    # print(sys_entro,highest_ig)
                    highest_ig_split =(uniq_elements[i],highest_ig,col_num)

            mapping_splits[str(index_cols_data[col_num])] = highest_ig_split
            breakpoint()
        else:
            # for now assuming the else block handles numerical continous columns
            # Greedy split
            cur_feat_col = cur_feat_col.astype(float)
            count_classes_left = [0]*num_classes
            sorted_index = np.argsort(cur_feat_col)
            # print(targ_col)
            sorted_feat, sorted_targ = cur_feat_col[sorted_index], targ_col[sorted_index]
            thresholds = []
            highest_ig_mean,highest_ig = (),0

            for i in range(len(cur_feat_col)-1):
                # print(targ_feats,sorted_targ[i])
                matched_ind_val = np.where(targ_feats == sorted_targ[i])[0][0]
                # print(matched_ind_val)
                count_classes_left[matched_ind_val] += 1

                if sorted_feat[i] == sorted_feat[i+1]:
                    continue

                mean =(sorted_feat[i] + sorted_feat[i+1])/2

                left_entro = _cal_entropy(_cal_probs(np.array(count_classes_left)))
                right_entro = _cal_entropy(_cal_probs(total_class_freq - np.array(count_classes_left)))
                
            
                total_left = sum(count_classes_left)
                total_right = num_rows -total_left
                p_left,p_right = total_left/num_rows, total_right/num_rows
                weighted_entro = _cal_weighted_entropy(np.array([p_left,p_right]),np.array([left_entro,right_entro]))
                # print(left_entro,right_entro)
                i_g = sys_entro - weighted_entro
                thresholds.append((mean,i_g))

                if i_g > highest_ig: 
                    # print(f"spotted highest entropy at {sorted_feat[i],sorted_feat[i+1]}")
                    highest_ig = i_g
                    # print(sys_entro,weighted_entro)
                    highest_ig_mean = (mean,highest_ig,col_num)
                
            # return highest_ig_mean
            mapping_splits[str(index_cols_data[col_num])] = highest_ig_mean
            breakpoint()
    mapping_splits = dict(sorted(mapping_splits.items(),key=lambda item:item[1][1],reverse=True))
    columns_gone.append(int(list(mapping_splits.keys())[0]))
    return mapping_splits


# support functions for calculating ig for numerical column
# find best split/ using histogram binding
def find_best_split(feat_cols_in:np.array,targ_col_in:np.array,cur_sys_entro = None, depth = 0)->float:
    """
    Find the best splitting criteria for a numerical column.

    Parameters:
    feat_cols, targ_col from cal_ig_col
    col_dtype : Data type of the column

    Reuturns:
    split_point: Best splitting criteria for the column.
    """
    # print(
        # f"Depth={depth}, "
    #     f"Samples={len(targ_col)}, "
    #     f"Entropy={cur_sys_entro:.3f}"
    # )
    global max_depth,num_cols,tree_formation
    # curr_depth = 0

    if cur_sys_entro == None:
        cur_sys_entro = _sys_entropy(targ_col,is_of=True)

    if abs(cur_sys_entro) < 1e-9 or feat_cols_in.shape[1] == 0:
        tree_formation[f'leaf_node_{depth}'] = targ_col_in
        return tree_formation
    

    # split_dict = dict(sorted(_cal_ig_col(feat_cols_in,targ_col_in,cur_sys_entro).items(),key=lambda item:item[1][1],reverse=True))
    split_dict = _cal_ig_col(feat_cols_in,targ_col_in,cur_sys_entro) 
    breakpoint()
    col_num, col_criteria,act_col = (list(split_dict.items())[0][0],list(split_dict.items())[0][1][0],list(split_dict.items())[0][1][2])
    index_cols_data.pop(index_cols_data.index(col_num))

    col_dtype = type_list[int(col_num)]

    if col_dtype == 'numeric':
        left_split_crit = (feat_cols_in[:,act_col]).astype(np.float64) <= col_criteria
        right_split_crit = ~left_split_crit

    else:
        left_split_crit = feat_cols_in[:,act_col] == col_criteria
        right_split_crit = ~left_split_crit
    
        
    cur_feat_cols = np.delete(feat_cols_in,act_col,axis=1)
    num_cols -= 1
    left_set_feats, right_set_feats = cur_feat_cols[left_split_crit], cur_feat_cols[right_split_crit]
    left_set_targ, right_set_targ = targ_col_in[left_split_crit], targ_col_in[right_split_crit]
    left_set_entro,right_set_entro = _sys_entropy(left_set_targ),_sys_entropy(right_set_targ)
    tree_formation[col_num] = col_criteria
    breakpoint()
    # left_best_split = 
    # prin(left_best_split)
    # right_best_split = 
    # print(right_best_split)
    return(find_best_split(left_set_feats,left_set_targ,left_set_entro,depth+1),find_best_split(right_set_feats,right_set_targ,right_set_entro,depth+1))

    # max_depth -=1

        
if __name__=="__main__":

    # making array
    # categorical_options = [['A','B','C','G'],['D','E','F']]
    # cat_cols = np.column_stack([np.random.choice(opt,size=30) for opt in categorical_options])
    # targ_col = np.random.choice([0,1,2],size=30)
    # num_cols = np.column_stack([np.random.randint(10,50,30),np.random.randint(20,40,30)])
    # data = np.column_stack([cat_cols,num_cols,targ_col])
    # print(data)
    data = np.array(
        [['G', 'D', '26', '26', '1'],
        ['A', 'E', '39', '32', '2'],
        ['G', 'F', '14', '38', '2'],
        ['G', 'E', '27', '31', '0'],
        ['G', 'E', '32', '39', '1'],
        ['G', 'E', '30', '23', '0'],
        ['G', 'D', '40', '33', '2'],
        ['C', 'F', '46', '29', '1'],
        ['B', 'D', '43', '28', '1'],
        ['G', 'F', '14', '31', '1'],
        ['G', 'D', '33', '29', '1'],
        ['A', 'D', '47', '26', '0'],
        ['B', 'F', '33', '24', '2'],
        ['G', 'E', '34', '20', '1'],
        ['B', 'F', '21', '29', '1'],
        ['B', 'F', '47', '25', '2'],
        ['C', 'E', '37', '28', '1'],
        ['C', 'D', '32', '37', '0'],
        ['C', 'E', '12', '31', '1'],
        ['B', 'F', '34', '20', '1'],
        ['B', 'D', '22', '35', '1'],
        ['B', 'F', '11', '25', '0'],
        ['A', 'E', '27', '34', '1'],
        ['B', 'F', '36', '23', '0'],
        ['C', 'F', '19', '21', '1'],
        ['G', 'D', '11', '32', '0'],
        ['B', 'F', '11', '33', '2'],
        ['G', 'E', '36', '38', '2'],
        ['A', 'E', '38', '33', '1'],
        ['C', 'D', '48', '28', '2']]
    )

    print(data)

    out = decision_tree_classify(data,params={'max_depth':3})
    # sorted_out = dict(sorted(out.items(),key=lambda item:item[1][1],reverse=True))

    # out1 = find_best_split(data[:,0],data[:,-1],out, col_dtype="U<21")
    print(out)

    # # debugging error in depth 3, where input is
    # problem_depth_in = np.array(
    #     [['E'],
    #    ['D'],
    #    ['E'],
    #    ['F']]
    # )

    # # input targ col for the depth 3
    # problem_depth_out = np.array(['1', '0', '1', '1'])

    # # give no splitting criteria/ dictionary
    # split_dict = dict(sorted(_cal_ig_col(problem_depth_in,problem_depth_out).items(),key=lambda item:item[1][1],reverse=True))
    # print(split_dict)
