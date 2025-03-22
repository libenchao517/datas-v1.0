################################################################################
## 本代码用于将RGBD数据集建模至Grassmann空间
################################################################################
## 导入模块
import os
import numpy as np
from random import sample
from DATA.utils import def_path
from DATA.utils import collect_images
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
# 解压数据
# import tarfile
# with tarfile.open('rgbd-dataset.tar', 'r') as tar:
#     tar.extractall(path='RGB-D')
# print("解压结束！")
################################################################################
## 定义基本变量
root_path = "rgbd-dataset"
target_dict = dict()
new_width = 20
new_height = 20
data = []
target = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
################################################################################
## 建模数据
categories = [item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item))]
for k, cate in enumerate(categories):
    target_dict[cate] = k
    cate_path = os.path.join(root_path, cate)
    objects = [item for item in os.listdir(cate_path) if os.path.isdir(os.path.join(cate_path, item))]
    for object in objects:
        object_path = os.path.join(cate_path, object)
        files = [item for item in os.listdir(object_path) if item.endswith("_crop.png")][:40]
        files = sample(files, 40)
        S = collect_images(path=object_path, file_list=files, new_width=new_width, new_height=new_height)
        data.append(S)
        target.append(target_dict.get(cate))
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "RGB-D-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
