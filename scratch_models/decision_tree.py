# from abc import ABC, abstractmethod

# class DecisionTree(ABC):
#     def __init__(self):
#         super().__init__()

# solo writing
import numpy as np
from collections import defaultdict
import math

def mimic_decisin_tree(tgc:np.array):
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
    
    
    uncertainity = np.sum(np.array(proba_cal) * np.log(proba_cal))

    # feature uncertainity calculation
    return proba_cal, f"uncertainity is {uncertainity}"
    
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


def histogram_binding(a:np.array):
    min, max = (np.min(a), np.max(a))
    k = int(np.log(a.size) + 1)
    step = (max - min)/k
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
            b.append((min_temp, up_e))

    return b


if __name__ == "__main__":
    test_arr = np.random.randint(10,59,30)
    print(f"{test_arr} \n min and max {(np.max(test_arr),np.min(test_arr))}")
    # out = mimic_decisin_tree(test_arr)
    out = histogram_binding(test_arr)
    print(out)