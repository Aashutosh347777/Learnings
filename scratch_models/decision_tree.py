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

def _cal_entropy():
    pass

    
def cumulative_linear_scan(arr_obj:np.array, targ:np.array,sys_entropy:float):
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

    total_pos_arr = np.unique(targ,return_counts=True)[1][1]
    total_rows = targ.shape[0]
    #iterating through the dict values
    left_count = list(freq_count.items())[0][1][0]
    left_pos = list(freq_count.items())[0][1][1]
    best_ig = 0
    for key,val in dict(list(freq_count.items())[1:-1]).items():
        left_count += val[0]
        left_pos += val[1]
        right_count = total_rows - left_count
        right_pos = total_pos_arr - left_pos

        p_left,p_right = left_pos/left_count, right_pos/right_count
        entropy = -((left_count/total_rows)*p_left* np.log(p_left) + (right_count/total_rows)*p_right * np.log(p_right))
        print("\n sys_entropy",sys_entropy," curr_entropy",entropy)
        ig = sys_entropy - entropy

        if ig > best_ig: best_ig = ig

    return (ig,key)
    

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


if __name__ == "__main__":
    test_arr = np.random.randint(10,59,30)
    y_test_arr = np.random.choice([0,1],size=30)
    print(y_test_arr)
    # print(f"{test_arr} \n min and max {(np.max(test_arr),np.min(test_arr))}")
    # out = mimic_decisin_tree(test_arr)
    out = cumulative_linear_scan(test_arr,y_test_arr,cal_uncert(y_test_arr))
    print(out)

# wrie the function to do things dynamically, may be it will be better if you make a new file