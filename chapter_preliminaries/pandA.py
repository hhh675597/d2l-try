import os

os.makedirs(os.path.join('', 'data'), exist_ok=True)
data_file = os.path.join('', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') #列名
    f.write('NA,pave,127500\n') #每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)

######练习######

na_counts = data.isna().sum() #计算每一列中缺失值的数量
col_to_drop = na_counts.idxmax()
print("删除缺失值最多的那一列:", col_to_drop)
data = data.drop(columns=[col_to_drop])
print(data)

###############
inputs, outputs = data.iloc[:, 0:1], data.iloc[:, 1]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
inputs = inputs.astype({col: 'int32' for col in inputs.columns if inputs[col].dtype == 'bool'})
print(inputs)
print(inputs.dtypes)

import torch
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)
