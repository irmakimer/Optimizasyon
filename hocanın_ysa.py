import numpy as np
import math
from datasetSpiral import ti,yi
#----------------------------------------
def exp(x):
    return np.array([math.exp(i) for i in x])
#-----------------------------------------
def tanh(x):
    if isinstance(x,float):
        result = (math.exp(x)-math.exp(-x))