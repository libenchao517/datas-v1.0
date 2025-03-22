from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
from pathlib import Path
from DATA.utils import data_intercept_for_mfd
mats = Path("MFPT").rglob("*.mat")
length = 1024
data_name = "MFPT-R"
os.makedirs("../MFD/" + data_name, exist_ok=True)
data, target = np.zeros((1, length)), np.zeros((1, 1))
for i, mat in enumerate(mats):
    temp = loadmat(mat)
    temp = temp.get('bearing')[0][0][2]
    temp_data = data_intercept_for_mfd(temp, sample_number=200, sample_size=length)
    data = np.concatenate((data, temp_data), axis=0)
    target = np.concatenate((target, (i + 1) * np.ones((len(temp_data), 1))), axis=0)
data, target = data[1:], target[1:]
pd.DataFrame(data).to_csv("../MFD/"+data_name+"/"+data_name+"_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/"+data_name+"/"+data_name+"_Target.csv", index=False, header=False)
