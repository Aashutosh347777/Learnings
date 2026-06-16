# Refactoring into functions

import numpy as np

def decision_tree_classify(dataset:np.array,params:dict):
    """
    Final call for getting the classification of the given data. Initializes the data and parameters for global access

    Parameters:
    dataset: A numpy array or a tabular data which the model gets trained on.
    params: Mathematical values required for building the model

    Returns:
    classification_model : A model to classify incoming values, trained on the params provided at the time of call
    """
    pass

def manage_data(dataset:np.array) -> tuple(np.array,np.array):
    """
    Ready the data for calculations.

    Parameters: 
    dataset: A numpy array or tabular data
    
    Returns
    targ_col: Separated target column.
    feat_cols: Separated feature columns.
    """

def _sys_entropy(targ_col:np.array) -> float:
    """
    Calculate system entropy.

    Parmeters:
    targ_column: The targetted column for classification

    Returns:
    sys_entro: Entropy value of the system.
    """


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

# support functions for calculating ig for numerical column
# find best split/ using histogram binding
def find_best_split(feat_col:np.array,targ_col:np.array,col_dtype:str) float:
    """
    Find the best splitting criteria for a numerical column.

    Parameters:
    feat_col, targ_col from cal_ig_col
    col_dtype : Data type of the column

    Reuturns:
    split_point: Best splitting criteria for the column.
    """

if __name__=="__main__":
    pass
