################################################################################
# 本代码用于整理CWRU轴承数据集
################################################################################
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from DATA.utils import Data_Intercept_for_MFD
os.makedirs("../MFD/CWRU", exist_ok=True)
order = ["CWRU/normal_0.mat", "CWRU/B007_0.mat", "CWRU/B014_0.mat",
         "CWRU/B021_0.mat", "CWRU/IR007_0.mat", "CWRU/IR014_0.mat",
         "CWRU/IR021_0.mat", "CWRU/OR007@6_0.mat", "CWRU/OR014@6_0.mat",
         "CWRU/OR021@6_0.mat"]
data, target = np.zeros((1, 1024)), np.zeros((1, 1))
for i, od in enumerate(order):
    temp = loadmat(od)
    k = [j for j in list(temp.keys()) if j.endswith("DE_time")]
    temp = temp[k[0]]
    temp_data = Data_Intercept_for_MFD(temp, sample_number=118, sample_size=1024)
    data = np.concatenate((data, temp_data), axis=0)
    target = np.concatenate((target, (i + 1) * np.ones((len(temp_data), 1))), axis=0)
data, target = data[1:], target[1:]
pd.DataFrame(data).to_csv("../MFD/CWRU/CWRU_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/CWRU/CWRU_Target.csv", index=False, header=False)
