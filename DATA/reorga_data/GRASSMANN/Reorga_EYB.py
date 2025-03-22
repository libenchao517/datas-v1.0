################################################################################
# 本文件用于整理Extended Yale B数据集至Grassmann空间
################################################################################
# 导入模块
import os
import re
import numpy as np
from DATA.utils import def_path
from DATA.utils import collect_images
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
# 定义基本变量
root_path = "ExtendedYaleB"
target_dict = dict()
new_width = 20
new_height = 20
data = []
target = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
################################################################################
# 建模数据
objects = [item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item))]
for k, object in enumerate(objects):
    object_path = os.path.join(root_path, object)
    object_files = [item for item in os.listdir(object_path) if re.match(r".*\d{2}.pgm", item)]
    target_dict[object] = k
    poses = np.unique([item.split("_")[1][:4] for item in object_files])
    for pose in poses:
        pose_files = [item for item in object_files if re.match(rf"{object}_{pose}.*.pgm", item)]
        S = collect_images(path=object_path, file_list=pose_files, new_width=new_width, new_height=new_height)
        data.append(S)
        target.append(target_dict.get(object))
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "EYB-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
