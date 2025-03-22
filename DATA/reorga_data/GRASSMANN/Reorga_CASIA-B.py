################################################################################
# 本文件用于整理CASIA-B数据集至Grassmann空间
################################################################################
# 导入模块
import os
import numpy as np
import pandas as pd
from DATA.utils import def_path
from DATA.utils import collect_images
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
# 定义基本变量
root_path = "CASIA-B"
target_dict = {"F" : 0, "M" : 1}
new_width = 20
new_height = 20
data = []
target = []
eta_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GS = GrassmannSubSpace()
GD = GrassmannDimensionality()
################################################################################
# 建模数据
label_path = "CASIA-B-subjects-info-pub.xls"
label = np.array(pd.read_excel(os.path.join(root_path, label_path)).iloc[:, 0:2])
label_dict = {i[0] : i[1] for i in label}
subjects = [item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item))]
for subject in subjects:
    subject_path = os.path.join(root_path, subject)
    gains = os.listdir(subject_path)
    for gain in gains:
        if not gain.endswith("01"):
            continue
        gait_path = os.path.join(subject_path, gain, "000")
        print(gait_path)
        files = [item for item in os.listdir(gait_path) if item.endswith(".png")]
        S = collect_images(path=gait_path, file_list=files, new_width=new_width, new_height=new_height)
        data.append(S)
        target.append(target_dict.get(label_dict.get(int(subject))))
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "CASIA-B-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
