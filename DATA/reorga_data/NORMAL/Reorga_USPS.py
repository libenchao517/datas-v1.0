################################################################################
# 本代码用于整理USPS手写数字数据集
################################################################################
import pandas as pd
from scipy.io import loadmat
USPS_path="./USPS/usps_all.mat"
data_path="./USPS/USPS_Data.csv"
target_path="./USPS/USPS_Target.csv"
data = loadmat(USPS_path)
data = data['data']
temp=[]
for i in range(10):
    temp.extend(data[:,:,i].T)
data=pd.DataFrame(temp)
target=[]
for i in range(1,11):
    temp=[i%10]*1100
    target.extend(temp)
target=pd.DataFrame(target)
data.to_csv(data_path,index=False,header=False)
target.to_csv(target_path,index=False,header=False)
