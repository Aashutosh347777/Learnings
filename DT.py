from abc import ABC
import numpy as np

# Gloabl definition
cols_dtype, cols_index = None, None

class _Helper():
    def __init__(s,feat_cols):
        """
        Initialize the Helper object for calculating the states that are needed for smooth functioning of the Decision Tree.

        Parameters:
            feat_cols: Numpy array whose columns nature are to be distinguished.
            num_cols : Number of columns in the given feature columns.
        """
        s.feat_cols = feat_cols
        s.num_cols = s.feat_cols.shape[1]


    def find_col_dtype(s)-> list:
        """
        Seperates the columns data type by thier nature continuous or discrete.

        Parameters:

        Returns:
        type_list : A list that specifies whether the columns are continous or discrete in nature. 
        """
        type_list = []
        for i in range(s.num_cols):
            # trying to convert the 1st element to float if success it is continous else discrete
            try:
                s.feat_cols[0,i]
                kind_conti = True
            except ValueError:
                kind_conti = False
            
            type_list.append('continuous' if kind_conti else 'discrete')
        
        return type_list
    
    def find_index_cols(s)->list:
        """
        Provide a list of indexes in string.

        Parameters:

        Returns:
        index_cols: A list of indexes as strings.
        """
        return [str(index) for index in range(s.num_cols)]

class DecisionTree(ABC):
    def __init__(s,feats:np.array,targ:np.array,params:dict):
        """
        Initialize the basic requirements to make a tree.

        Parameters:
            feats: Feature column.
            targ: Target column.
            params: Tree parameters.
            cols_index: Current state of the available columns.
            cols_dtype: Current nature of the available columns.
        """
        s.feats = feats
        s.targ = targ
        s.params = params
        s.sys_entro = None
        
        helper = _Helper(s.feats)
        s.cols_dtype = helper.find_col_dtype()
        s.cols_index = helper.find_index_cols()

        global cols_dtype, cols_index
        cols_dtype,cols_dtype = s.cols_dtype.copy(), s.cols_index.copy()


    def _build_tree(s):
        """
        Recursively build the tree by adding nodes.

        Parameters:

        Returns:
            Node: A Node object can either be Decision Node or Leaf Node.
        """
        pass


class Node(ABC):
    def __init__(s,cur_sys_entro:float,cur_feat_cols:np.array,cur_targ_col:np.array,index_mapping:list):
        """
        Initialize the basic requirments for a node.

        Parameters:
            cur_sys_entro: calculated system entropy.
            cur_feat_cols: Column whoose Infromation gain is to be calculate.
            cur_targ_col_in: Target column of the set.
            index_mapping: State of current left index
        
        Returns:
        """
        global cols_dtype, cols_index
        glob_cols_dtype, glob_cols_index = cols_dtype, cols_index

        s.curs_sys_entro,s.cur_feat_cols,s.cur_targ_col, s.index_mapping = cur_sys_entro, cur_feat_cols, cur_targ_col, index_mapping


    def _find_best_column_criteria(s):
        """
        Find the best column for the next split.

        Params:

        Returns:
        """
        pass
    

class DecisionTreeClassify(DecisionTree):
    def __init__(s):
        """
        Initialize Classification variable that might be needed
        """
        super().__init__()            

        # cols_dtype = DecisionTree.cols_dtypev
    
class DecisionTreeRegressor(DecisionTree):
    def __init__(s):
        super().__init__()

class DecisionNode(Node):
    def __init__(s,left_node:Node,right_node:Node,criteria:dict):
        """
        Initialize the Decision Node with criterai to seperate left and right of the tree.
        """
        super().__init__()
        s.left_node = left_node
        s.right_node = right_node
        s.criteria = criteria

    def _find_best_column_criteria(s,index_mapping):
        """
        Calculate weighted entopy for colummns based on the data type categorical or numerical.

        Parameters:
            feat_cols_in: Column whoose Infromation gain is to be calculate.
            targ_col_in: Target column of the set.
            sys_entro: calculated system entropy.

        Returns:
            col_ig: Information gained of the feature column.
        """
      
        if index_mapping is None:
            index_mapping = s.glob_cols_index

        for col_num in range(len(index_mapping)):
            cur_feat_col = s.cur_feat_cols[:,col_num]
            cur_col_dtype = s.glob_cols_dtype[col_num]


class LeafNode(Node):
    def __init__(s,value):
        """
        Intialize the Leaf node with the predicted target value.
        """
        super().__init__()
        s.value = value

class _CriteriaFinder:
    # def __init__(s,feats_in:np.array,targ_in:np.array,sys_entro):
    #     s.feats_in, s.targ_in,s.sys_entro = feats_in, targ_in, sys_entro

    def _cal_probs(feat_col,return_metrics:bool=False,is_num_cols:bool=False)->np.array:
        """
        Calculated probability for a catergorically sperateable cols.

        Parameters:
        arr_obj: A numpy array object.
        feat_col: A selected column.

        Returns:
        probs: A numpy array object containing probability for unique elements.
        """
        var_Counts = np.unique(feat_col,return_counts=True)
        uniq_eles,freq_count = var_Counts

        if return_metrics:
            return(freq_count/np.sum(freq_count),uniq_eles,freq_count)
        
        # doubtable
        if is_num_cols:
            return feat_col/np.sum(feat_col)

        return freq_count/np.sum(freq_count)

    def _cal_entropy(s,feat_col)->float:
        """
        Calculate the entropy for a given probabilites.

        Parameters:
        feat_col: A selected column.

        Returns:
        entropy: Entropy/ unecertaininty in the system
        """
        probs = s._cal_probs(feat_col)
        return -sum(probs*np.log2(probs))
    
    def _cal_weighted_entropy(s,probs,entropies)->float:
        """
        Calculate the weighted entropy of given probability and entropy pairs.

        Parameters:
        
        Returns:
        weighted_entropy: Weighted for a particular state of a column.
        """
        return sum(probs * entropies)
    
