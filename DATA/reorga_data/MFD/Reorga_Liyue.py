################################################################################
# 本代码用于整理Liyue轴承数据集
################################################################################
import os
import numpy as np
import pandas as pd
from DATA.utils import Data_Intercept_for_MFD
os.makedirs("../MFD/Liyue", exist_ok=True)
order = ["B6", "C6", "O6", "B20", "C20", "O20", "B35", "C35", "O35"]
temp = pd.read_csv("Liyue/NM.csv", header=None, index_col=None).to_numpy()
data = Data_Intercept_for_MFD(temp, sample_number=200, sample_size=1024)
target = np.ones((len(data), 1))
for i, od in enumerate(order):
    temp = pd.read_csv("Liyue/" + od+".csv", header=None, index_col=None).to_numpy()
    temp_data = Data_Intercept_for_MFD(temp, sample_number=200, sample_size=1024)
    data = np.concatenate((data, temp_data), axis=0)
    target = np.concatenate((target, (i+2)*np.ones((len(temp_data), 1))), axis=0)
pd.DataFrame(data).to_csv("../MFD/Liyue/Liyue_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/Liyue/Liyue_Target.csv", index=False, header=False)
