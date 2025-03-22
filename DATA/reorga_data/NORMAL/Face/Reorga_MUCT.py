################################################################################
# 本文件用于整理MUCT数据集
################################################################################
# 导入模块
import os
import re
import shutil
import numpy as np
from PIL import Image
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_data_name
from DATA.utils import save_data_labels
from icecream import ic
################################################################################
# 定义基本变量
root_path = "muct-master"
data_name = "MUCT"
target = []
pixel_matrix = []
################################################################################
# 建模数据
# files = [item for item in os.listdir(root_path)]
# for num in range(625):
#     num_str = "i{:03d}".format(num)
#     subjects = [file for file in files if file.startswith(num_str)]
#     if subjects:
#         os.makedirs(os.path.join(root_path, num_str), exist_ok=True)
#         for sub in subjects:
#             from_path = os.path.join(root_path, sub)
#             to_path = os.path.join(root_path, num_str)
#             shutil.move(from_path, to_path)

data_path, target_path, target_dict_path = def_path(data_name, option="NORMAL", file_type=".csv")
categories = [item for item in os.listdir(root_path)]
target_dict = {b : a for a, b in enumerate(categories)}
for cate in categories:
    cate_path = os.path.join(root_path, cate)
    files = [item for item in os.listdir(cate_path)]
    for file in files:
        file_path = os.path.join(cate_path, file)
        image = Image.open(file_path)
        image = image.resize((28, 28))
        image = image.convert("L")
        pixel_vector = np.array(image.getdata())
        pixel_matrix.append(pixel_vector)
        target.append(target_dict.get(cate))
data = np.array(pixel_matrix)
target = np.array(target)
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_data_labels(data_name=data_name, abbre_labels=categories)
save_data_name(options="NORMAL", data_name=data_name)
ic(data.shape, target.shape)
