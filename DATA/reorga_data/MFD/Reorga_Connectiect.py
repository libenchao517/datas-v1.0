################################################################################
# 本代码用于整理Connectiect齿轮数据集
################################################################################
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
os.makedirs("../MFD/Connectiect", exist_ok=True)
Data = loadmat("Connectiect/DataForClassification_TimeDomain .mat")
data = Data['AccTimeDomain'].T
data = data[:, :1024]
target = np.array(range(len(data)))
target //= 104
target +=1
pd.DataFrame(data).to_csv("../MFD/Connectiect/Connectiect_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/Connectiect/Connectiect_Target.csv", index=False, header=False)

# Number of gear fault types=9
# {'healthy','missing','crack','spall',
# 'chip5a','chip4a','chip3a','chip2a','chip1a'}
# Number of samples per type=104
# Number of total samples=9x104=936
# The data are collected in sequence,
# the first 104 samples are healthy,
# 105th ~208th samples are missing, and etc.
