################################################################################
# 本代码用于整理letter字母数据集
################################################################################
import pandas as pd
letter_path="./letter/letter-recognition.data"
data_path = './letter/letter_Data.csv'
target_path = './letter/letter_Target.csv'
letter = pd.read_table(letter_path, header=None, delimiter=',')
letter_data = letter.iloc[:, 1:]
letter_target = letter.iloc[:, 0].transform(func=lambda x: ord(x) - 64)
letter_data.to_csv(data_path,index=False,header=False)
letter_target.to_csv(target_path,index=False,header=False)
