import os
import numpy as np
import pandas as pd
from DATA.utils import save_data_name
data_health = np.zeros((1, 1024))
data_inner = np.zeros((1, 1024))
data_outer = np.zeros((1, 1024))
data_combina = np.zeros((1, 1024))
data_roller = np.zeros((1, 1024))
################################################################################
#  整理Ottawa数据集
dn = "Ottawa"
Dp = os.path.join("..", "MFD", dn, dn + "_Data.csv")
Tp = os.path.join("..", "MFD", dn, dn + "_Target.csv")
data_B1 = pd.read_csv(Dp, header=None, index_col=None).to_numpy()
target_B1 = pd.read_csv(Tp, header=None, index_col=None).to_numpy()
temp = data_B1[(target_B1 == 1).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 164)
data_health = np.concatenate((data_health, temp[sorts]), axis=0)
temp = data_B1[(target_B1 == 2).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 145)
data_inner = np.concatenate((data_inner, temp[sorts]), axis=0)
temp = data_B1[(target_B1 == 3).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 188)
data_outer = np.concatenate((data_outer, temp[sorts]), axis=0)
################################################################################
#  整理Polito数据集
dn = "Polito"
Dp = os.path.join("..", "MFD", dn, dn + "_Data.csv")
Tp = os.path.join("..", "MFD", dn, dn + "_Target.csv")
data_B2 = pd.read_csv(Dp, header=None, index_col=None).to_numpy()
target_B2 = pd.read_csv(Tp, header=None, index_col=None).to_numpy()
temp = data_B2[(target_B2 == 1).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 64)
data_health = np.concatenate((data_health, temp[sorts]), axis=0)
temp = data_B2[(target_B2 == 2).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 57)
data_inner = np.concatenate((data_inner, temp[sorts]), axis=0)
temp = data_B2[(target_B2 == 3).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 57)
data_inner = np.concatenate((data_inner, temp[sorts]), axis=0)
temp = data_B2[(target_B2 == 5).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 80)
data_roller = np.concatenate((data_roller, temp[sorts]), axis=0)
temp = data_B2[(target_B2 == 6).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 80)
data_roller = np.concatenate((data_roller, temp[sorts]), axis=0)
################################################################################
#  整理Sebear数据集
dn = "Sebear"
Dp = os.path.join("..", "MFD", dn, dn + "_Data.csv")
Tp = os.path.join("..", "MFD", dn, dn + "_Target.csv")
data_B3 = pd.read_csv(Dp, header=None, index_col=None).to_numpy()
target_B3 = pd.read_csv(Tp, header=None, index_col=None).to_numpy()
temp = data_B3[(target_B3 == 1).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 272)
data_health = np.concatenate((data_health, temp[sorts]), axis=0)
temp = data_B3[(target_B3 == 2).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 340)
data_roller = np.concatenate((data_roller, temp[sorts]), axis=0)
temp = data_B3[(target_B3 == 3).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 500)
data_combina = np.concatenate((data_combina, temp[sorts]), axis=0)
temp = data_B3[(target_B3 == 4).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 241)
data_inner = np.concatenate((data_inner, temp[sorts]), axis=0)
temp = data_B3[(target_B3 == 5).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 312)
data_outer = np.concatenate((data_outer, temp[sorts]), axis=0)

#  合并数据
new_data= np.concatenate((data_health[1:], data_inner[1:]), axis=0)
new_data= np.concatenate((new_data, data_outer[1:]), axis=0)
new_data= np.concatenate((new_data, data_combina[1:]), axis=0)
new_data= np.concatenate((new_data, data_roller[1:]), axis=0)

from icecream import ic
ic(
    data_health.shape,
    data_inner.shape,
    data_outer.shape,
    data_roller.shape,
    data_combina.shape
)

new_target = np.ones((500, 1))
for i in range(4):
    new_target = np.concatenate((new_target, (i + 2) * np.ones((500, 1))), axis=0)
new_target = new_target.flatten()
new_dn = "Mix-Bear"
new_Dp = os.path.join("..", "MFD", new_dn, new_dn + "_Data.csv")
new_Tp = os.path.join("..", "MFD", new_dn, new_dn + "_Target.csv")
os.makedirs(os.path.join("..", "MFD", new_dn), exist_ok=True)
save_data_name("MFD", new_dn)
pd.DataFrame(new_data).to_csv(new_Dp, index=False, header=False)
pd.DataFrame(new_target).to_csv(new_Tp, index=False, header=False)
