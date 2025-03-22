################################################################################
# 本文件用于实现加载数据集的标准化
################################################################################
# 导入模块
import json
from pathlib import Path
import numpy as np
import pandas as pd
import medmnist as mm
from sklearn.preprocessing import MinMaxScaler
from .Preprocessing import Pre_Procession as pp
################################################################################
# 加载数据类
class Load_Data():
    # 初始化部分
    def __init__(
            self,
            data_name,
            is_scaler = True
    ):
        """
        初始化函数
        :param data_name: 数据集名称
        :param is_scaler: 是否进行标准化
        """
        self.PATH = "/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1])
        self.data_name = data_name
        self.Select_Class()
        self.start_text = "当前正在加载" + data_name + "数据集......"
        self.end_text = "\r" + data_name + "数据集加载完毕！" + " " * 10
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_scaler = is_scaler

    def Select_Class(self):
        """
        类别选择函数
        根据data_options.json文件确定数据集所在的文件夹
        :return: 确定数据集的文件夹self.opt_class
        """
        root = Path(__file__).parts[:-1]
        path_names = "/".join(list(root) + ["data_options.json"])
        with open(path_names, 'r', encoding='utf-8') as names:
            OPT = json.load(names)
        names.close()
        for opt in OPT.keys():
            if self.data_name in OPT.get(opt):
                self.opt_class = opt
                break
        if self.opt_class in ["GRASSMANN", "THREEDIMEN" , "SPD"]:
            self.data_path = "_Data.npy"
            self.target_path = "_Target.npy"
        elif self.opt_class.startswith("MEDMNIST"):
            self.data_path = self.PATH + "/DATA" + "/MEDICAL"
        else:
            self.data_path = "_Data.csv"
            self.target_path = "_Target.csv"

    def Loading(self):
        """
        加载常规数据的主函数
        :return: 数据，标签
        """
        print(self.start_text, end="")
        Dp = self.PATH + "/DATA" + "/" + self.opt_class + "/" + self.data_name + "/" + self.data_name + self.data_path
        Tp = self.PATH + "/DATA" + "/" + self.opt_class + "/" + self.data_name + "/" + self.data_name + self.target_path
        if self.opt_class in ["GRASSMANN", "THREEDIMEN" , "SPD"]:
            data = np.load(Dp)
            target = np.load(Tp)
        else:
            data = pd.read_csv(Dp, header=None, index_col=None)
            target = pd.read_csv(Tp, header=None, index_col=None)
            # MFD中的数据在预处理时已经进行了标准化
            if self.opt_class != "MFD" and self.is_scaler:
                data = self.scaler.fit_transform(data)
        print(self.end_text)
        return np.array(data), np.array(target).reshape(-1)

    def Load_lgd(self):
        """
        加载GRASSMANN或SPD类型的数据的标签字典并用作绘图师的图例
        其它类型的数据也可以从data_labels_abbre.json文件中加载
        :return: 标签字典
        """
        Lp = self.PATH + "/DATA" + "/" + self.opt_class + "/" + self.data_name + "/" + self.data_name + "_Target_Dict.json"
        if self.opt_class in ["GRASSMANN", "THREEDIMEN", "SPD"]:
            with open(Lp, 'r', encoding='utf-8') as paras:
                lgds = json.load(paras)
            paras.close()
        return lgds

    def Load_MedMNIST(self):
        """
        加载MedMNIST数据集
        :return: 数据，标签
        """
        print(self.start_text, end="")
        info = mm.INFO[self.data_name]
        DataClass = getattr(mm.dataset, info['python_class'])
        train = DataClass(split='train', root=self.data_path)
        test = DataClass(split='test', root=self.data_path)
        val = DataClass(split='val', root=self.data_path)
        data = np.concatenate((train.imgs, test.imgs, val.imgs))
        data = data.reshape((data.shape[0], -1))
        target = np.concatenate((train.labels, test.labels, val.labels))
        if self.data_name == "pathmnist":
            data, target = self._Load_PathMNIST(data, target)
        if self.data_name == "chestmnist":
            data, target = self._Load_ChestMNIST(data, target)
        if self.is_scaler:
            data = self.scaler.fit_transform(data)
        print(self.end_text)
        return data, target.reshape(-1)

    def _Load_PathMNIST(self, data, target):
        """
        私有函数
        由于PathMNIST很大，则固定从每类抽取1500个样本
        :param data: Load_MedMNIST函数提供的完整数据
        :param target: Load_MedMNIST函数提供的完整标签
        :return: 子集数据，子集标签
        """
        data, _, target, _ = pp().uniform_sampling(data, target, train_size=1500, random_state=42)
        return data, target

    def _Load_ChestMNIST(self, data, target):
        """
        私有函数
        ChestMNIST数据集是多标签数据集
        本函数选取了所有只患单种疾病的样本
        :param data: Load_MedMNIST函数提供的完整数据
        :param target: Load_MedMNIST函数提供的完整标签
        :return: 子集数据，子集标签
        """
        temp = np.sum(target, axis=1)
        index_0 = temp == 0 # 健康索引
        index_1 = temp == 1 # 单疾病索引
        data_0 = data[index_0] # 健康数据
        data_1, target_1 = data[index_1], target[index_1] # 单疾病数据
        target_0 = np.zeros((len(data_0), 1)) # 健康标签
        # 单疾病标签
        target_1 = np.array([np.array(i).argmax() for i in target_1]).reshape((-1, 1))
        data = np.concatenate((data_0, data_1), axis=0)
        target = np.concatenate((target_0, target_1), axis=0)
        return data, target
