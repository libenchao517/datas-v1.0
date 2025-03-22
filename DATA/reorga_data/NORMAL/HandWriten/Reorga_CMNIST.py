################################################################################
# 本文件用于整理Chinese MNIST数据集
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
root_path = "ChineseMNIST/data"
data_name = "CMNIST"
target = []
pixel_matrix = []
labels = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿"]
target_dict = {a : b for a, b in enumerate(labels)}
################################################################################
# 建模数据
data_path, target_path, target_dict_path = def_path(data_name, option="NORMAL", file_type=".csv")
files = [item for item in os.listdir(root_path) if item.endswith(".jpg")]
for file in files:
    file_path = os.path.join(root_path, file)
    image = Image.open(file_path)
    image = image.resize((28, 28))
    image = image.convert("L")
    pixel_vector = np.array(image.getdata())
    pixel_matrix.append(pixel_vector)
    temp = int((file.split("_")[-1]).split(".")[0])
    target.append(temp)
data = np.array(pixel_matrix)
target = np.array(target) - 1
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_data_labels(data_name=data_name, abbre_labels=labels)
save_data_name(options="NORMAL", data_name=data_name)
ic(data.shape, target.shape)
