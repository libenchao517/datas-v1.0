################################################################################
# 本代码用于整理SouthEast University Gear齿轮数据集
################################################################################
import os
import numpy as np
import pandas as pd
from DATA.utils import Data_Intercept_for_MFD
os.makedirs("../MFD/Segear", exist_ok=True)
def process(name):
    print("开始处理"+name)
    Data = pd.read_csv("Segear/"+ name +"_20_0.csv", delimiter='\t', header=None, index_col=None)
    Data = Data.map(lambda x : x.split("\t"))
    Data = pd.DataFrame(Data[0].apply(pd.Series).values)
    Data = Data.iloc[:, :-1]
    Data = Data.map(lambda x : float(x))
    Data = Data.to_numpy()
    return Data
order = ["Health", "Chipped", "Miss", "Root", "Surface"]
data, target = np.zeros((1, 1024)), np.zeros(((1, 1)))
for i, od in enumerate(order):
    Data = process(od)
    Data = Data_Intercept_for_MFD(Data[:, 1:], sample_number=500, sample_size=1024)
    data = np.concatenate((data, Data), axis=0)
    target = np.concatenate((target, (i + 1) * np.ones((len(Data), 1))), axis=0)
data, target = data[1:], target[1:]
pd.DataFrame(data).to_csv("../MFD/Segear/Segear_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/Segear/Segear_Target.csv", index=False, header=False)
