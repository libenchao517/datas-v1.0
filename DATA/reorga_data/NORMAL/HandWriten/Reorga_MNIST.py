############e####################################################################
# 本代码用于整理MNIST手写数字数据集
################################################################################
import idx2numpy
import numpy as np
from DATA.utils import def_path
from DATA.utils import save_data
from DATA.utils import save_data_name
from DATA.utils import save_data_labels
from icecream import ic
################################################################################
# 定义基本变量
data_name = "MNIST"
data_path, target_path, target_dict_path = def_path(data_name, option="NORMAL", file_type=".csv")
labels = [i for i in range(10)]
target_dict = {a : b for a, b in enumerate(labels)}
################################################################################
# 建模数据
data_train = idx2numpy.convert_from_file("MNIST/train-images.idx3-ubyte").reshape([60000, 784])
data_test= idx2numpy.convert_from_file("MNIST/t10k-images.idx3-ubyte").reshape([10000, 784])
target_train = idx2numpy.convert_from_file("MNIST/train-labels.idx1-ubyte")
target_test= idx2numpy.convert_from_file("MNIST/t10k-labels.idx1-ubyte")
data = np.concatenate((data_train, data_test), axis=0)
target = np.concatenate((target_train, target_test), axis=0)
save_data(data, target, target_dict, data_path, target_path, target_dict_path)
save_data_labels(data_name=data_name, abbre_labels=labels)
save_data_name(options="NORMAL", data_name=data_name)
ic(data.shape, target.shape)
