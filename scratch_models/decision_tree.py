# from abc import ABC, abstractmethod

# class DecisionTree(ABC):
#     def __init__(self):
#         super().__init__()

# solo writing
import numpy as np
from collections import defaultdict
import math


def format_for_calcu(tgc:np.array):
    counts_val = np.unique(tgc,return_counts=True)
    total = np.sum(counts_val[1])

    refined_keys = [str(key) for key in counts_val[0].astype(int).astype(str)]

    zipped_item = zip(refined_keys,counts_val[1])
    dict_obj = dict(zipped_item)

    #calculate probability
    x = dict_obj.values()
    proba_cal = [a/total for a in x]
    
    for key, new_vals in zip(dict_obj.keys(), proba_cal):
        dict_obj[key] = new_vals
    
    return proba_cal

def cal_uncert(tgc:np.array):
    proba_cal = format_for_calcu(tgc)
    uncertainity = -(np.sum(np.array(proba_cal) * np.log2(proba_cal)))
    # feature uncertainity calculation
    return uncertainity

# def cal_log_of_proba(arr_obj:np.array, targ:np.array):
#     freq_count = histogram_binding(arr_obj,targ)

def _cal_entropy(probs:np.array):
    return -sum(list(map(lambda x: x*np.log2(x),probs)))

def _cal_probs(arr_obj:np.array)->np.array:
    total_rows = arr_obj.shape[0]
    counts = np.unique(arr_obj,return_counts=True)
    freq_count = counts[1]

    probs = freq_count/np.sum(freq_count)
    return probs

def _cal_weighted_entropy(probs:np.array, entropies:np.array):
    return sum(probs * entropies)

    
def cumulative_linear_scan(arr_obj:np.array, targ:np.array,sys_entropy:float):
    min, max = (np.min(arr_obj), np.max(arr_obj))
    k = int(np.log(arr_obj.size) + 1)
    step = (max - min)/k
    b = []
    freq_count = {}
    up_e = 0
    # total_pos_arr = 0
    for i in range(k):
        # if i == 0:
        up_e = min+step
        b.append((min, up_e)) 
        mask = (arr_obj>=min) & (arr_obj<up_e)
        # count = np.sum(np.where((arr_obj>=min)&(arr_obj<up_e),1,0))
        # indexes = np.where((arr_obj>=min)&(arr_obj<up_e))[0]
        curr_elemenst = arr_obj[mask]
        unique_ele,count_ele = np.unique(targ[mask],return_counts=True)
        
        # if unique_ele[0].size > 1:
        #     total_pos = unique_ele[1][1]
        # else:
        #     total_pos = 0
        # # total__pos_arr += total_pos
        freq_count[(min,up_e)] = (len(unique_ele),count_ele)
        min = up_e

    ig_and_bins, best_ig = [], 0
    # left_bin_entro
    left_bin_counts = list(freq_count.items())[0][1][1]
    right_bin_counts = list(freq_count.items())[0][1][1]
    print("Pringint left bins---\n")
    print(left_bin_counts)


    for key,val in dict(list(freq_count.items())[1:-1]).items():
        probs_bin =_cal_probs(val[1])
        entorpy_bin = _cal_entropy(probs_bin)

        ig = sys_entropy - entorpy_bin

        ig_and_bins.append((key,ig))

        if ig > best_ig: 
            best_ig = ig
            print(f"----Best I.G---- {best_ig}")

    # total_pos_arr = np.unique(targ,return_counts=True)[1][1]
    # total_rows = targ.shape[0]
    # #iterating through the dict values
    # left_count = list(freq_count.items())[0][1][0]
    # left_pos = list(freq_count.items())[0][1][1]
    # best_ig = 0
    # for key,val in dict(list(freq_count.items())[1:-1]).items():
    #     left_count += val[0]
    #     left_pos += val[1]
    #     right_count = total_rows - left_count
    #     right_pos = total_pos_arr - left_pos

    #     p_left,p_right = left_pos/left_count, right_pos/right_count
    #     entropy = -((left_count/total_rows)*p_left* np.log(p_left) + (right_count/total_rows)*p_right * np.log(p_right))
    #     print("\n sys_entropy",sys_entropy," curr_entropy",entropy)
    #     ig = sys_entropy - entropy

    #     if ig > best_ig: best_ig = ig

    return ig_and_bins
    # return (ig,key)
    
def greedy_split(arr_obj:np.array, targ:np.array,sys_entropy:float):
    # target column assessment:
    num_rows = targ.shape[0]
    targ_stats = np.unique(targ_col,return_counts=True)
    targ_feats, num_classes,total_class_freq = targ_stats[0],len(targ_stats[0]), targ_stats[1]
    count_classes_left = [0]*num_classes
    sorted_index = np.argsort(arr_obj)

    sorted_feat, sorted_targ = arr_obj[sorted_index], targ[sorted_index]
    thresholds = []
    highest_ig_mean,highest_ig = (),0
    counter = 0
    print(sorted_feat)
    for i in range(len(arr_obj)-1):
        matched_ind_val = np.where(targ_feats ==sorted_targ[i])[0][0]
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

        i_g = sys_entropy - weighted_entro
        thresholds.append((mean,i_g))

        if i_g > highest_ig: 
            print(f"spotted highest entropy at {sorted_feat[i],sorted_feat[i+1]}")
            highest_ig = i_g
            highest_ig_mean = (highest_ig, mean)

        counter += 1    
    
    print(counter)    
    return highest_ig_mean


    

def finding_best_splits(ftc:np.array):
    num_rows, num_cols = ftc.shape
    for i in range(num_clos):
        min, max = (np.min(ftc[:,i]), np.max(ftc[:,i]))
        k = int(np.log(ftc[:,i].size) + 1)
        step_size = (max - min)/k
        b = []
        up_e = 0
        min_temp = 0
        for i in range(k):
            if i == 0:
                up_e = min+step
                b.append((min, min+step)) 
            else:
                min_temp = up_e
                up_e = min_temp + step
                b.append(min_temp, up_e)
        
    return b


def histogram_binding(arr_obj:np.array,targ:np.array):
    min, max = (np.min(arr_obj), np.max(arr_obj))
    k = int(np.log(arr_obj.size) + 1)
    step = (max - min)/k
    b = []
    freq_count = {}
    up_e = 0
    min_temp = 0
    # total_pos_arr = 0
    for i in range(k):
        # if i == 0:
        up_e = min+step
        b.append((min, up_e)) 
        count = np.sum(np.where((arr_obj>=min)&(arr_obj<up_e),1,0))
        indexes = np.where((arr_obj>=min)&(arr_obj<up_e))[0]
        total_pos = np.unique(targ[indexes],return_counts=True)[1][1]
        # total__pos_arr += total_pos
        freq_count[(min,up_e)] = (count,total_pos)
        min = up_e


        # else:
        #     min_temp = up_e
        #     up_e = min_temp + step
        #     b.append((min_temp, up_e))
        #     count = np.sum(np.where((a>=min_temp) & (a<up_e), 1,0))

        #     indexes = np.where((a>=min_temp)&(a<up_e))[0]
        #     total_pos = np.unique(c[indexes],return_counts=True)[1][1]
        #     freq_count[(min_temp,up_e)] = (count,total_pos)
        
    return freq_count

def best_split_categorical(col_obj:np.array,targ:np.array) -> tuple:
    sys_entro = _cal_entropy(_cal_probs(targ))
    print(sys_entro )
    #assume that we know the unique elements in target column and list of uniq values
    targ_list_val = [0,1,2]
    targ_uniq_count = 3
    probs = _cal_probs(col_obj)
    sum_probs = sum(probs)
    probs_per_obj = []
    
    total_rows=col_obj.shape[0]
    uniq_val_counts=np.unique(col_obj,return_counts=True)
    uniq_ele, freq_uniq_ele = uniq_val_counts
    count_uniq_ele = uniq_val_counts[1].size
    highest_ig_split,highest_ig = (),0
    for i in range(count_uniq_ele):
        probs_per_obj = [probs[i],sum_probs-probs[i]]
        cur_ele = uniq_ele[i]
        #binary masking
        mask1 = col_obj==cur_ele
        # indexes,indexes_other_ele=np.where(col_obj==cur_ele)[0],np.where(col_obj!=cur_ele)[0]
        indexes,indexes_other_ele=col_obj[mask1],col_obj[~mask1]
        # assume binary calssification
        correspondent_vals,correspondent_vals_other_ele=targ[mask1],targ[~mask1]
        total_curr_ele_count=correspondent_vals.size
        ## binary masking
        total_other_ele_count=total_rows-total_curr_ele_count

        print("---For the curr_element---\n")
        print(np.column_stack([indexes,correspondent_vals]))

        print("\n---For the curr_element---\n")
        print(np.column_stack([indexes_other_ele,correspondent_vals_other_ele]))

        # pos_val_count,pos_val_count_other_ele=sum(correspondent_vals==1),sum(correspondent_vals_other_ele==1)

        # num_rows_ele,num_rows_other_than_ele = correspondent_vals.shape[0],correspondent_vals_other_ele.shape[0]

        # cur_ele_counts,other_ele_counts = np.unique(correspondent_vals,return_counts=True), np.unique(correspondent_vals_other_ele,return_counts=True)

        # # for cur ele
        # proba_curr_ele_in = cur_ele_counts[1]/num_rows_ele
        # proba_other_ele_in = other_ele_counts[1]/num_rows_other_than_ele

        proba_curr_ele_in,proba_other_ele_in = _cal_probs(correspondent_vals),_cal_probs(correspondent_vals_other_ele)

        entroyp_curr_ele_in, entropy_other_ele_in = _cal_entropy(proba_curr_ele_in), _cal_entropy(proba_other_ele_in)

        # store_probs= []
        # for vals in targ_list_val:
        #     if correspondent_vals==vals:
        #         proba_ith_uniq_ele = sum(correspondent_vals==vals)/total_curr_ele_count
            
        #     if correspondent_vals_other_ele==vals:
        #         proba_ith_remain_ele = sum(correspondent_vals_other_ele==vals)/total_curr_ele_count

        #     store_probs.append((proba_ith_uniq_ele,proba_ith_remain_ele))


        # proba_curr_ele,proba_remain_ele=pos_val_count/total_curr_ele_count,pos_val_count_other_ele/total_other_ele_count
        # for 
        # entroyp_curr_ele = _cal_entropy((proba_curr_ele,1-proba_curr_ele))
        # entropy_remain_ele = _cal_entropy((proba_remain_ele,1-proba_remain_ele))
        
        entropies = np.array([entroyp_curr_ele_in,entropy_other_ele_in])
        curr_ig = sys_entro-_cal_weighted_entropy(probs_per_obj,entropies)
        print(f"IG {curr_ig}")
        if curr_ig > highest_ig: 
            highest_ig = curr_ig            
            highest_ig_split =(uniq_ele[i],highest_ig)

    return highest_ig_split

def find_next_split_col(arr_obj:np.array)-> dict:
    feat_cols,targ_col = arr_obj[:,:-1],arr_obj[:,-1]
    total_rows,total_cols = feat_cols.shape

    sys_entro = _cal_entropy(_cal_probs(targ_col))
    criteria = {}

    for i in range(total_cols):
        col1 = feat_cols[:,i]
        if col1.dtype=='<U21':
            pair = best_split_categorical(col1,targ_col)
            print(pair)
            criteria[str(pair[0])] =pair[1]
        else:
            cumulative_linear_scan(col1,targ_col,sys_entro)
            criteria[str(pair[0])] =pair[1]
    
    return criteria

# How Categorical Data Uses Extension ArraysThe pandas.Categorical type is one of the most common implementations of an Extension Array. Instead of forcing string objects into a slow, resource-heavy NumPy object array, pandas splits categorical columns into two distinct, memory-efficient NumPy arrays:Categories Array: A unique lookup table contaiing the distinct string labels (e.g., ['high', 'low', 'medium']).Codes Array: An array of small integers (e.g., [0, 1, 2, 0]) that map back to the indexes of the categories array.
def find_cols_type(arr:np.array)->list:
    type_list = []
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


if __name__ == "__main__":
    # test_arr = np.random.randint(10,59,30)
    # y_test_arr = np.random.choice([0,1],size=30)
    # test_arr_categoricl = np.random.choice(['A','B','C'],size=30)
    # print(y_test_arr)
    # print(test_arr_categoricl)
    # # print(f"{test_arr} \n min and max {(np.max(test_arr),np.min(test_arr))}")
    # # out = mimic_decisin_tree(test_arr)
    # # out = cumulative_linear_scan(test_arr,y_test_arr,cal_uncert(y_test_arr))
    # # print(out)

    # print(best_split_categorical(test_arr_categoricl,y_test_arr))

    categorical_options = [['A','B','C'],['D','E','F']]
    cat_cols = np.column_stack([np.random.choice(opt,size=30) for opt in categorical_options])
    targ_col = np.random.choice([0,1,2],size=30)
    num_cols = np.column_stack([np.random.randint(10,50,30),np.random.randint(20,40,30)])
    data = np.column_stack([cat_cols,num_cols,targ_col])
    # print(data)

    # print(find_next_split_col(data))
    # print(best_split_categorical(cat_cols[:,0],targ_col))

    # print("\n---Input col----\n")
    # # print(data[:,2].astype(np.int64))
    # print(greedy_split(data[:,2].astype(np.int64),data[:,-1].astype(np.int64),_cal_entropy(_cal_probs(data[:,-1]))))

    print(find_cols_type(data))


# wrie the function to do things dynamically, may be it will be better if you make a new file