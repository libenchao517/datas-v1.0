################################################################################
# 本代码用于整理CoverType数据集
################################################################################
import gzip
import pandas as pd
CT_path = "./CoverType/covtype.data.gz"
data_path="./CoverType/CoverType_Data.csv"
target_path="./CoverType/CoverType_Target.csv"
with gzip.open(CT_path, "rb") as file:
    CT_data = file.read()
CT_data=CT_data.decode()
CT_data=CT_data.split("\n")
CT_data= CT_data[:-1]
re_CT_data=[]
for i in CT_data:
   re_CT_data.append(i.split(','))
re_CT_data=pd.DataFrame(re_CT_data)
data=pd.DataFrame(re_CT_data.iloc[:, 0:54]).to_csv(data_path, index=False, header=False)
target=pd.DataFrame(re_CT_data.iloc[:, 54]).to_csv(target_path, index=False, header=False)
