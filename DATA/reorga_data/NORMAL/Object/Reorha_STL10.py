################################################################################
# 本文件用于整理STL-10数据集
################################################################################
# 导入模块
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_data_name
from DATA.utils import save_data_labels
from icecream import ic
################################################################################
# 定义基本变量
data_name = "STL-10"
pixel_matrix = []
labels = [item for item in range(10)]
target_dict = {b : a for a, b in enumerate(labels)}
train = loadmat("stl10_matlab/train.mat")
test = loadmat("stl10_matlab/test.mat")
data = np.concatenate((train.get("X"), test.get("X")), axis=0)
target = np.concatenate((train.get("y"), test.get("y")), axis=0).flatten()
target = target - 1
################################################################################
# 建模数据
data_path, target_path, target_dict_path = def_path(data_name, option="NORMAL", file_type=".csv")
data = np.uint8(data)
data = data.reshape((data.shape[0], 3, 96, 96))
data = data.transpose((0, 2, 3, 1))
for d in data:
    image = Image.fromarray(d, mode="RGB")
    image = image.resize((28, 28))
    image = image.convert("L")
    pixel_vector = np.array(image.getdata())
    pixel_matrix.append(pixel_vector)
data = np.array(pixel_matrix)
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_data_labels(data_name=data_name, abbre_labels=labels)
save_data_name(options="NORMAL", data_name=data_name)
ic(data.shape, target.shape)
