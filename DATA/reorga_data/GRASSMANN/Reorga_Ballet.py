################################################################################
# 本文件用于整理Ballet数据集至Grassmann空间
################################################################################
# 导入模块
import os
import numpy as np
import scipy
from DATA.utils import def_path
from DATA.utils import collect_images
from DATA.utils import save_data
from DATA.utils import save_grassmann_paras
from Grassmann import GrassmannSubSpace
from Grassmann import GrassmannDimensionality
################################################################################
# 定义基本变量
root_path = "ballet"
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
ballet = scipy.io.loadmat("ballet/labels.mat")
act_labels = ballet.get("act_labels")
labels = ballet.get("labels").flatten()
for k, act in enumerate(act_labels[0]):
    target_dict[act[0]] = k
sequences = [item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item))]
for seq, label in zip(sequences, labels):
    seq_path = os.path.join(root_path, seq)
    files = [item for item in os.listdir(seq_path) if item.endswith(".jpg")]
    start = 0
    end = 0
    t = label[0]
    while end < len(label):
        if t != label[end]:
            S = collect_images(path=seq_path, file_list=files[start : end], new_width=new_width, new_height=new_height)
            data.append(S)
            target.append(t-1)
            t = label[end]
            start = end
        end += 1
    S = collect_images(path=seq_path, file_list=files[start: end], new_width=new_width, new_height=new_height)
    data.append(S)
    target.append(t-1)
################################################################################
# 建模导Grassmann流形
images = data
for p in range(5, 16):
    print("Modeling Grassmann", p)
    data_name = "Ballet-" + str(p)
    data_path, target_path, target_dict_path = def_path(data_name, option="GRASSMANN")
    data = GS.compute_subspace(images, p=p)
    target = np.array(target)
    save_data(data, target, target_dict, data_path, target_path, target_dict_path)
    save_grassmann_paras(data_name=data_name, grassmann_p=p, picture_height=new_height, picture_width=new_width)
    for eta in eta_list:
        GD.ratio = eta
        GD.determine_dimensionality(data)
        GD.save_low_dimensions(data_name=data_name)
