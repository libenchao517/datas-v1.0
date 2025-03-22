import os
import numpy as np
import pandas as pd
from DATA.utils import save_data_name

length = 128
for dn in [
    "CWRU", "JiangNan", "Liyue", "Connectiect", "Ottawa", "Sebear",
    "Segear", "Polito", "WuHan", "KAT", "SCA", "MFPT-R"]:
    Dp = os.path.join("..", "MFD", dn, dn + "_Data.csv")
    Tp = os.path.join("..", "MFD", dn, dn + "_Target.csv")
    new_dn = dn + "-" + str(length)
    os.makedirs(os.path.join("..", "MFD", new_dn), exist_ok=True)
    new_Dp = os.path.join("..", "MFD", new_dn, new_dn + "_Data.csv")
    new_Tp = os.path.join("..", "MFD", new_dn, new_dn + "_Target.csv")
    data = pd.read_csv(Dp, header=None, index_col=None).to_numpy()
    target = pd.read_csv(Tp, header=None, index_col=None).to_numpy()
    new_data = data.reshape((-1, 128))
    new_target = np.tile(target, (1, 8)).reshape((-1, 1))
    save_data_name("MFD", new_dn)
    pd.DataFrame(new_data).to_csv(new_Dp, index=False, header=False)
    pd.DataFrame(new_target).to_csv(new_Tp, index=False, header=False)
