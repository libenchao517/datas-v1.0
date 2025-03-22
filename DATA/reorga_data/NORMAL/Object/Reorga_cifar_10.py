################################################################################
# 本文件用于整理CIFAR 10数据集
################################################################################
# 导入模块
import os
import numpy as np
import pickle
from PIL import Image
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_data_name
from DATA.utils import save_data_labels
from icecream import ic
################################################################################
# 定义基本变量
root_path = "cifar-10-python/cifar-10-batches-py"
data_name = "CIFAR-10"
pixel_matrix = []
labels = [str(item) for item in range(10)]
target_dict = {b : a for a, b in enumerate(labels)}
################################################################################
# 建模数据
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file_path = os.path.join(root_path, "test_batch")
bin = unpickle(file_path)
data, target = np.array(bin[b"data"]), np.array(bin[b"labels"])

for i in range(5):
    file_path = os.path.join(root_path, "data_batch_"+str(i+1))
    bin = unpickle(file_path)
    data_, target_ = np.array(bin[b"data"]), np.array(bin[b"labels"])
    data = np.concatenate((data, data_), axis=0)
    target = np.concatenate((target, target_), axis=0)

data = np.uint8(data)
data = data.reshape((data.shape[0], 3, 32, 32))
data = data.transpose((0, 2, 3, 1))

data_path, target_path, target_dict_path = def_path(data_name, option="NORMAL", file_type=".csv")

for d in data:
    image = Image.fromarray(d, mode="RGB")
    # image = image.resize((28, 28))
    image = image.convert("L")
    pixel_vector = np.array(image.getdata())
    pixel_matrix.append(pixel_vector)
data = np.array(pixel_matrix)
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_data_labels(data_name=data_name, abbre_labels=labels)
save_data_name(options="NORMAL", data_name=data_name)
ic(data.shape, target.shape)
