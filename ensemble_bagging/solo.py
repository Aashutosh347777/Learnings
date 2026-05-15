import numpy as np
# Data bootstrapping
n_models = 0

#dummy numpy data
coeff = np.random.randn(100000,5)
const = np.random.rand(100000,1)
var = np.linspace(-5,5,100000).reshape(100000,1)

dum_arr = var
for i in range(2,6):
    upd_var = np.power(var,i)
    final_var_arr = np.hstack((upd_var,dum_arr))
    dum_arr = final_var_arr

data = np.hstack((coeff * final_var_arr, const))

print("\nShape",data.shape)

def create_data(data:np.array):
    """Create data for Bagging ensembling."""
    pass


if __name__=="__main__":
    print("In main section.")

