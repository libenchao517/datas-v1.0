################################################################################
# 本代码用于整理Ottawa轴承数据集
################################################################################
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from DATA.utils import Data_Intercept_for_MFD
os.makedirs("../MFD/Ottawa", exist_ok=True)
order = ["I", "O"]
temp = loadmat("Ottawa/H-A-1.mat")
data = Data_Intercept_for_MFD(temp["Channel_1"], sample_number=300, sample_size=1024)
target = np.ones((len(data), 1))
for i, od in enumerate(order):
    temp = loadmat("Ottawa/" + od+"-A-1.mat")
    temp_data = Data_Intercept_for_MFD(temp["Channel_1"], sample_number=300, sample_size=1024)
    data = np.concatenate((data, temp_data), axis=0)
    target = np.concatenate((target, (i+2)*np.ones((len(temp_data), 1))), axis=0)
pd.DataFrame(data).to_csv("../MFD/Ottawa/Ottawa_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/Ottawa/Ottawa_Target.csv", index=False, header=False)
