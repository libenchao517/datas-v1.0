import os
import numpy as np
import pandas as pd
from DATA.utils import save_data_name
data_health = np.zeros((1, 1024))
data_misstooth = np.zeros((1, 1024))

################################################################################
#  整理Connectiect数据集
dn = "Connectiect"
Dp = os.path.join("..", "MFD", dn, dn + "_Data.csv")
Tp = os.path.join("..", "MFD", dn, dn + "_Target.csv")
data_G1 = pd.read_csv(Dp, header=None, index_col=None).to_numpy()
target_G1 = pd.read_csv(Tp, header=None, index_col=None).to_numpy()

temp = data_G1[(target_G1 == 1).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 50)
data_health = np.concatenate((data_health, temp[sorts]), axis=0)
temp = data_G1[(target_G1 == 2).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 50)
data_misstooth = np.concatenate((data_misstooth, temp[sorts]), axis=0)

temp = data_G1[(target_G1 == 3).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_crack = temp[sorts]
temp = data_G1[(target_G1 == 4).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_spall = temp[sorts]
temp = data_G1[(target_G1 == 5).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_chip5a = temp[sorts]
temp = data_G1[(target_G1 == 6).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_chip4a = temp[sorts]
temp = data_G1[(target_G1 == 7).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_chip3a = temp[sorts]
temp = data_G1[(target_G1 == 8).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_chip2a = temp[sorts]
temp = data_G1[(target_G1 == 9).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_chip1a = temp[sorts]
################################################################################
#  整理Segear数据集
dn = "Segear"
Dp = os.path.join("..", "MFD", dn, dn + "_Data.csv")
Tp = os.path.join("..", "MFD", dn, dn + "_Target.csv")
data_G2 = pd.read_csv(Dp, header=None, index_col=None).to_numpy()
target_G2 = pd.read_csv(Tp, header=None, index_col=None).to_numpy()

temp = data_G2[(target_G2 == 1).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 50)
data_health = np.concatenate((data_health, temp[sorts]), axis=0)
temp = data_G2[(target_G2 == 3).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 50)
data_misstooth = np.concatenate((data_misstooth, temp[sorts]), axis=0)
temp = data_G2[(target_G2 == 2).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_chipped = temp[sorts]
temp = data_G2[(target_G2 == 4).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_root = temp[sorts]
temp = data_G2[(target_G2 == 5).flatten()]
sorts = np.random.choice(np.arange(len(temp)), 100)
data_surface = temp[sorts]

from icecream import ic
ic(
    data_health.shape,
    data_misstooth.shape,
    data_chip1a.shape,
    data_chip2a.shape,
    data_chip3a.shape,
    data_chip4a.shape,
    data_chip5a.shape,
    data_chipped.shape,
    data_surface.shape,
    data_root.shape,
    data_spall.shape,
    data_crack.shape
)

new_data= np.concatenate((data_health[1:], data_misstooth[1:]), axis=0)
new_data= np.concatenate((new_data, data_spall), axis=0)
new_data= np.concatenate((new_data, data_crack), axis=0)
new_data= np.concatenate((new_data, data_root), axis=0)
new_data= np.concatenate((new_data, data_surface), axis=0)
new_data= np.concatenate((new_data, data_chipped), axis=0)
new_data= np.concatenate((new_data, data_chip1a), axis=0)
new_data= np.concatenate((new_data, data_chip2a), axis=0)
new_data= np.concatenate((new_data, data_chip3a), axis=0)
new_data= np.concatenate((new_data, data_chip4a), axis=0)
new_data= np.concatenate((new_data, data_chip5a), axis=0)

ic(new_data.shape)

new_target = np.ones((100, 1))
for i in range(11):
    new_target = np.concatenate((new_target, (i + 2) * np.ones((100, 1))), axis=0)
new_target = new_target.flatten()

new_dn = "Mix-Gear"
new_Dp = os.path.join("..", "MFD", new_dn, new_dn + "_Data.csv")
new_Tp = os.path.join("..", "MFD", new_dn, new_dn + "_Target.csv")
os.makedirs(os.path.join("..", "MFD", new_dn), exist_ok=True)
save_data_name("MFD", new_dn)
pd.DataFrame(new_data).to_csv(new_Dp, index=False, header=False)
pd.DataFrame(new_target).to_csv(new_Tp, index=False, header=False)
