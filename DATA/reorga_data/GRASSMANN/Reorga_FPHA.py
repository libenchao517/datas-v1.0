################################################################################
## 本代码用于将FPHA数据集建模至Grassmann空间
################################################################################
## 导入模块
import os
import numpy as np
import pandas as pd
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
## 定义基本变量
root_path = "Hand_pose_annotation_v1"
target_dict = dict()
new_width = 0
new_height = 0
data = []
target = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
################################################################################
## 建模数据
for i in range(6):
    subject_path = os.path.join(root_path, "Subject_" + str(i+1))
    categories = os.listdir(subject_path)
    for k, cate in enumerate(categories):
        if (cate, k) not in target_dict.items():
            target_dict[cate] = k
        cate_path = os.path.join(subject_path, cate)
        objects = [item for item in os.listdir(cate_path) if os.path.isdir(os.path.join(cate_path, item))]
        for object in objects:
            object_path = os.path.join(cate_path, object, "skeleton.txt")
            try:
                S = pd.read_csv(object_path, delimiter=" ")
                data.append(S.iloc[:, 1:].T.to_numpy())
                target.append(target_dict.get(cate))
            except:
                pass
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "FPHA-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
