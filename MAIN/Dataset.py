import torch
import torch.utils.data.dataset as Dataset
import numpy as np

#创建子类
class CustomDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label

class CustomDataset1(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data1, Data2, Label):
        self.Data1 = Data1
        self.Data2 = Data2
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data1)
    #得到数据内容和标签
    def __getitem__(self, index):
        data1 = torch.Tensor(self.Data1[index])
        data2 = torch.Tensor(self.Data2[index])
        label = torch.Tensor(self.Label[index])
        return data1, data2, label