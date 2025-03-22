################################################################################
# 本文件用于整理KTH TIPS数据集
################################################################################
# 导入模块
import os
import numpy as np
from PIL import Image
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_data_name
from DATA.utils import save_data_labels
from icecream import ic
################################################################################
# 定义基本变量
root_path = "KTH_TIPS"
data_name = "KTH-TIPS"
target = []
pixel_matrix = []
labels = categories = [item for item in os.listdir(root_path)]
target_dict = {b : a for a, b in enumerate(labels)}
################################################################################
# 建模数据
data_path, target_path, target_dict_path = def_path(data_name, option="NORMAL", file_type=".csv")
files = [item for item in os.listdir(root_path) if item.endswith(".jpg")]
for cate in categories:
    cate_path = os.path.join(root_path, cate)
    files = [item for item in os.listdir(cate_path) if item.endswith(".png")]
    for file in files:
        file_path = os.path.join(cate_path, file)
        image = Image.open(file_path)
        image = image.resize((28, 28))
        image = image.convert("L")
        pixel_vector = np.array(image.getdata())
        pixel_matrix.append(pixel_vector)
        target.append(target_dict.get(cate))
data = np.array(pixel_matrix)
target = np.array(target) - 1
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_data_labels(data_name=data_name, abbre_labels=labels)
save_data_name(options="NORMAL", data_name=data_name)
ic(data.shape, target.shape)
