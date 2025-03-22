################################################################################
## 本代码用于将Virus数据集建模至Grassmann空间
################################################################################
## 导入模块
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from DATA.utils import def_path
from DATA.utils import collect_images
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
## 定义基本变量
root_path = "Virus/Virus-Texture-original-16bit"
target_dict = dict()
new_width = 20
new_height = 20
data = []
target = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
kf = KFold(n_splits=10, shuffle=True)
################################################################################
## 建模数据
temp = pd.read_csv("Virus/classNames.csv", delimiter=";").to_numpy()
for t in temp:
    target_dict[t[1]] = t[0] - 1
for i in range(15):
    files = [item for item in os.listdir(root_path) if re.match("class-" + "{:03d}".format(i+1) + "-sample*", item)]
    files = np.array(files)
    for train_index, test_index in kf.split(files):
        _, files_ = files[train_index], files[test_index]
        S = collect_images(path=root_path, file_list=files_, new_width=new_width, new_height=new_height)
        data.append(S)
        target.append(i)
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "Virus-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
