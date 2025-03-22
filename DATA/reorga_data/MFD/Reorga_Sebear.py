################################################################################
# 本代码用于整理SouthEast University Bearing轴承数据集
################################################################################
import os
import numpy as np
import pandas as pd
from DATA.utils import Data_Intercept_for_MFD
os.makedirs("../MFD/Sebear", exist_ok=True)
def process(name):
    print("开始处理"+name)
    Data = pd.read_csv("Sebear/"+ name +"_20_0.csv", delimiter='\t', header=None, index_col=None)
    Data = Data.map(lambda x : x.split("\t"))
    Data = pd.DataFrame(Data[0].apply(pd.Series).values)
    Data = Data.iloc[:, :-1]
    Data = Data.map(lambda x : float(x))
    Data = Data.to_numpy()
    return Data
order = ["comb", "inner", "outer"]
data = process("health")
data = Data_Intercept_for_MFD(data, sample_number=500, sample_size=1024)
target = np.ones(((len(data), 1)))
print("开始处理"+"ball")
Data = pd.read_csv("Sebear/ball_20_0.csv").to_numpy()
Data = Data_Intercept_for_MFD(Data, sample_number=500, sample_size=1024)
data = np.concatenate((data, Data), axis=0)
target = np.concatenate((target, 2*np.ones((len(Data), 1))))
for i, od in enumerate(order):
    Data = process(od)
    Data = Data_Intercept_for_MFD(Data, sample_number=500, sample_size=1024)
    data = np.concatenate((data, Data), axis=0)
    target = np.concatenate((target, (i + 3) * np.ones((len(Data), 1))), axis=0)
pd.DataFrame(data).to_csv("../MFD/Sebear/Sebear_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/Sebear/Sebear_Target.csv", index=False, header=False)
