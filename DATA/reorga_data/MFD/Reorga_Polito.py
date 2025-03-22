################################################################################
# 本代码用于整理Polito轴承数据集
################################################################################
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from DATA.utils import Data_Intercept_for_MFD
os.makedirs("../MFD/Polito", exist_ok=True)
mat_list = ["Polito/C0A_100_000_1.mat", "Polito/C1A_100_000_2.mat",
            "Polito/C2A_100_000_1.mat", "Polito/C3A_100_000_1.mat",
            "Polito/C4A_100_000_1.mat", "Polito/C5A_100_000_1.mat",
            "Polito/C6A_100_000_1.mat"]
data, target = np.zeros((1, 1024)), np.zeros((1, 1))
for m, mat in enumerate(mat_list):
    temp = loadmat(mat)
    temp = temp[mat.split("/")[-1][:-4]]
    temp_data = Data_Intercept_for_MFD(temp, sample_number=118, sample_size=1024)
    data = np.concatenate((data, temp_data), axis=0)
    target = np.concatenate((target, (m + 1) * np.ones((len(temp_data), 1))), axis=0)
data, target = data[1:], target[1:]
pd.DataFrame(data).to_csv("../MFD/Polito/Polito_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/Polito/Polito_Target.csv", index=False, header=False)
