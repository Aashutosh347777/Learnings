# Refactoring into functions

import numpy as np

# Global vars, constructor assignment
num_rows:int=0
num_cols:int=0
num_classes:int=0

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
    manage_data(dataset)
    return _sys_entropy()

def manage_data(dataset:np.array) -> None:
    """
    Ready the data for calculations.

    Parameters: 
    dataset: A numpy array or tabular data. Sets global variables.
    
    Returns
    targ_col: Separated target column.
    feat_cols: Separated feature columns.
    """
    global targ_col,feat_col,num_rows,num_cols

    # dataset dimesion and allocation
    num_rows,num_cols = dataset.shape
    feat_cols,targ_col = dataset[:,:-1],dataset[:,-1]

    # for now return None
    return None

def _cal_probs(arr_obj:np.array)->np.array:
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
    return freq_count/num_rows

def _cal_entropy(probs:np.array)-> float:
    """
    Calculate the entropy for a given probabilites.

    Parameters:
    probs: Porbability for each class.

    Returns:
    entropy: Entropy/ unecertaininty in the system
    """
    return -sum(probs*np.log2(probs))

def _sys_entropy() -> float:
    """
    Calculate system entropy.

    Parmeters:
    targ_column: The targetted column for classification

    Returns:
    sys_entro: Entropy value of the system.
    """
    return _cal_entropy(_cal_probs(targ_col))


def cal_ig_col(feat_col:np.array,targ_col:np.array,sys_entro:float)-> float:
    """
    Calculate weighted entopy for colummns based on the data type categorical or numerical.

    Parameters:
    feat_col: Column whoose Infromation gain is to be calculate.
    targ_col: Target column of the set.
    sys_entro: calculated system entropy.

    Returns:
    col_ig: Information gained of the feature column.
    """
    pass

# support functions for calculating ig for numerical column
# find best split/ using histogram binding
def find_best_split(feat_col:np.array,targ_col:np.array,col_dtype:str)->float:
    """
    Find the best splitting criteria for a numerical column.

    Parameters:
    feat_col, targ_col from cal_ig_col
    col_dtype : Data type of the column

    Reuturns:
    split_point: Best splitting criteria for the column.
    """
    pass

if __name__=="__main__":
    # making array
    categorical_options = [['A','B','C'],['D','E','F']]
    cat_cols = np.column_stack([np.random.choice(opt,size=30) for opt in categorical_options])
    targ_col = np.random.choice([0,1,2],size=30)
    num_cols = np.column_stack([np.random.randint(10,50,30),np.random.randint(20,40,30)])
    data = np.column_stack([cat_cols,num_cols,targ_col])

    out = decision_tree_classify(data,{})
    print(out)