import numpy as np
def transform(data):
    for i in data.columns:
        data[i+'_sqrt']=np.sqrt(data[i])