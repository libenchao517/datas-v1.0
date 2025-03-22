import json
from pathlib import Path
from .Load import Load_Data
from .Preprocessing import Pre_Procession
root = Path(__file__).parts[:-1]

## datas中列举了所有可以加载的数据
path_names = "/".join(list(root) + ["data_options.json"])
with open(path_names, 'r', encoding='utf-8') as names:
    datas = json.load(names)
names.close()

## abbre_labels中列举了部分数据集的标签集合
path_labels = "/".join(list(root) + ["data_labels_abbre.json"])
with open(path_labels, 'r', encoding='utf-8') as labels:
    abbre_labels = json.load(labels)
labels.close()
