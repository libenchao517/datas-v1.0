################################################################################
# 本代码用于整理KAT轴承数据集
################################################################################
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from DATA.utils import Data_Intercept_for_MFD
os.makedirs("../MFD/KAT", exist_ok=True)
order = ["K001", "KA01", "KB23", "KI01"]
data, target = np.ones((1, 1024)), np.ones((1, 1))
for i, od in enumerate(order):
    name = "N09_M07_F10_" + od + "_1"
    temp = loadmat("KAT/" + name + ".mat")[name][0][0][2][-1][-1][2].reshape((-1, 1))
    temp_data = Data_Intercept_for_MFD(temp, sample_number=200, sample_size=1024)
    data = np.concatenate((data, temp_data), axis=0)
    target = np.concatenate((target, (i+1)*np.ones((len(temp_data), 1))), axis=0)
data, target = data[1:], target[1:]
pd.DataFrame(data).to_csv("../MFD/KAT/KAT_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/KAT/KAT_Target.csv", index=False, header=False)
