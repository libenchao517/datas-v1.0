################################################################################
# 本文件用于整理Kannada MNIST数据集
################################################################################
# 导入模块
import numpy as np
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_data_name
from DATA.utils import save_data_labels
from icecream import ic
################################################################################
# 定义基本变量
data_name = "KaMNIST"
data_path, target_path, target_dict_path = def_path(data_name, option="NORMAL", file_type=".csv")
labels = [i for i in range(10)]
target_dict = {a : b for a, b in enumerate(labels)}
################################################################################
# 建模数据
data_train = np.load("KannadaMNIST/Kannada_MNIST/X_kannada_MNIST_train.npz")["arr_0"].reshape([60000, 784])
data_test = np.load("KannadaMNIST/Kannada_MNIST/X_kannada_MNIST_test.npz")["arr_0"].reshape([10000, 784])
target_train = np.load("KannadaMNIST/Kannada_MNIST/y_kannada_MNIST_train.npz")["arr_0"]
target_test = np.load("KannadaMNIST/Kannada_MNIST/y_kannada_MNIST_test.npz")["arr_0"]
data = np.concatenate((data_train, data_test), axis=0)
target = np.concatenate((target_train, target_test), axis=0)
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_data_labels(data_name=data_name, abbre_labels=labels)
save_data_name(options="NORMAL", data_name=data_name)
ic(data.shape, target.shape)
