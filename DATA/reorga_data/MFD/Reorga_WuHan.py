################################################################################
# 本代码用于整理WuHan University Rotor转子数据集
################################################################################
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
os.makedirs("../MFD/WuHan", exist_ok=True)
data = loadmat("WuHan/180data_new_select_denoised.mat")['Y_wavedeno'][:, :1024]
target = np.arange(180) // 45 + 1
pd.DataFrame(data).to_csv("../MFD/WuHan/WuHan_Data.csv", index=False, header=False)
pd.DataFrame(target).to_csv("../MFD/WuHan/WuHan_Target.csv", index=False, header=False)

# These data are denoised signals processed by wavelet
# thresholding-based denoising.
# They are represented by Data_size 2-dimensional matrix.
# Each row represents Data_size vibration signal,
# and the first 45 rows belong to normal rotor,
# the second contact-rubbing,
# the third unbalance and the final misalignment.
# Each column represents the length of data, 2048, or time, 1s.
