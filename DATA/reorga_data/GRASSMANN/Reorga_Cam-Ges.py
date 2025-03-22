################################################################################
# 本文件用于整理Cambridge Gesture数据集至Grassmann空间
################################################################################
# 导入模块
import numpy as np
import scipy
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
# 定义基本变量
data_name = "Cam-Ges-6"
root_path = "Cam-Ges"
data_path, target_path, target_dict_path = def_path(data_name)
target_dict = {i : i-1 for i in range(1, 10)}
new_width = 20
new_height = 20
p = 6
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
################################################################################
# 建模数据
print("Modeling Grassmann", 6)
cam = scipy.io.loadmat("Cam-Ges/Cambridge.mat").get("grassSet")
data = np.transpose(cam[0][0][0], axes=(2, 0, 1))
target = np.array(cam[0][0][1] - 1).flatten()
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
for eta in eta_list:
    GD.ratio = eta
    GD.determine_dimensionality(data)
    GD.save_low_dimensions(data_name=data_name)
