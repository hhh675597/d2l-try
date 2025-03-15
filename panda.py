import os

os.makedirs(os.path.join('', 'data'), exist_ok=True)
#os.path.join('', 'data') 将空字符串和 'data' 连接成路径
#第一个参数 '' 是空字符串，表示当前目录
#第二个参数 'data' 是目标文件夹名
#结果会生成相对路径 './data'
#exist_ok=True 表示如果目录已存在也不会报错
data_file = os.path.join('', 'data', 'house_tiny.csv')
#os.path.join() 这里有三个参数:
#第一个参数 '' 同样是空字符串，表示当前目录
#第二个参数 'data' 是文件夹名
#第三个参数 'house_tiny.csv' 是文件名
#最终会生成路径 './data/house_tiny.csv'
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') #列名
    f.write('NA,pave,127500\n') #每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd #这个python程序不能命名为pandas.py,否则编译器会报错AttributeError: partially initialized module 'pandas' has no attribute 'read_csv' (most likely due to a circular import)

data = pd.read_csv(data_file)
print(data)

#处理缺失值，插值法
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True)) #本地vscode上运行时需要添加numeric_noly=True,否则会编译错误
#错误原因：当某列包含字符串时，mean() 无法计算均值。常见情况：数据文件中有缺失值（如空字符串或 "NaN" 文本），导致列被识别为 object 类型
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
#inputs = inputs.astype({col: 'int32' for col in inputs.columns if inputs[col].dtype == 'bool'})
print(inputs)
print(inputs.dtypes) 
#检查inputs所有列的数据类型.输出为下面四行
#NumRooms      float64
#Alley_pave       bool
#Alley_nan        bool
#dtype: object(只是说明这个 Series 用来存储各列类型的数据类型是 object,前三列的确均是数值类型)

#现在inputs和outputs所有条目都是数值类型，可以转换为张量格式
import torch
import numpy
x, y = torch.tensor(inputs.astype(numpy.float32).values), torch.tensor(outputs.astype(numpy.float32).values)

print(x, y)
#?报错：can't convert np.ndarray of type numpy.object_.整个DataFrame的底层NumPy数组的统一dtype为object
#虽然float32和bool类型各自在tensor转换范围内，但这两个不同的类型使得DataFrame的类型变为object
#解决方案：添加强制转换类型