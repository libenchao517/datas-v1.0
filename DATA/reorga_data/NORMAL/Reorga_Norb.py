################################################################################
# 本代码用于整理Norb玩具齿轮数据集
################################################################################
import pandas as pd
train_data_path = './Norb/train_data_1.csv'
test_data_path = './Norb/test_data_1.csv'
train_target_path = './Norb/train_target.csv'
test_target_path = './Norb/test_target.csv'
data_path = './Norb/Norb_Data.csv'
target_path = './Norb/Norb_Target.csv'
train_data = pd.read_csv(train_data_path, header=None)
test_data = pd.read_csv(test_data_path, header=None)
train_target = pd.read_csv(train_target_path, header=None).iloc[:, 0]
test_target = pd.read_csv(test_target_path, header=None).iloc[:, 0]
data = pd.concat([train_data, test_data], axis=0)
target = pd.concat([train_target, test_target], axis=0)
data.to_csv(data_path,index=False,header=False)
target.to_csv(target_path,index=False,header=False)
