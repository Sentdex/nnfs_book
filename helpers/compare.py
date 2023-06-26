import numpy as np



def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
