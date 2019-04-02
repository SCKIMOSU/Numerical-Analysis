import numpy as np
from scipy.optimize import fsolve

fm=lambda m: np.sqrt(9.81*m/0.25)*np.tanh(np.sqrt(9.81*0.25/m)*4)-36
m=fsolve(fm, 1)
print("Real Root= ", m)
