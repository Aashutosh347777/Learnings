from abc import ABC
import numpy as np

# class Dataset():
#     def __init__(self,dataset:np.array):
#         self.feats:np.array = dataset[:,:-1]
#         self.targ:np.array = dataset[:,-1]
#         self.type_feats:list = None
#         self.index_feat_cols:list = None
    
#     def find_state():
#         pass

class DecisionTree(ABC):
    def __init__(self,feats:np.array,targ:np.array,params:dict):
        """
        Initialize the basic requirements to make a tree.
        """
        self.feats = feats
        self.targ = targ
        self.params = params
    
    


class DecisionTreeClassify(DecisionTree):
    def __init__(self):
        super().__init__()

class DecisionTreeRegressor(DecisionTree):
    def __init__(self):
        super().__init__()

class _CriteriaFinder():
    def __init__(self):
        super().__init__()