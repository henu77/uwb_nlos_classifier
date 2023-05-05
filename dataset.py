import os
import numpy as np
import paddle
from paddle.io import Dataset


def import_uwb_data_from_files_all(root_dir):
    """
        Read .csv files and store data into an array
        format: |LOS|NLOS|data...|
    """
    output_arr = []
    first = 1
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            filename = os.path.join(dirpath, file)
            output_data = []
            # read data from file
            input_data = np.loadtxt(fname=filename, dtype="float", delimiter=',', skiprows=1)
            # append to array
            if first > 0:
                first = 0
                output_arr = input_data
            else:
                output_arr = np.vstack((output_arr, input_data))
            print(filename, '已加载')
    return output_arr

class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """

    def __init__(self, data_dir, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset, self).__init__()
        uwb_data = import_uwb_data_from_files_all(root_dir=data_dir)
        # get CIR of single preamble pulse
        for item in uwb_data:
            item[15:] = item[15:] / float(item[2])
        # 删除第1-14列
        data = np.delete(arr=uwb_data, obj=range(1, 15), axis=1)
        data_arr = data[:, 1:]
        min_values = data_arr.min(axis=1)
        max_values = data_arr.max(axis=1)
        data_arr -= min_values[:, np.newaxis]  # 将最小值沿着列方向进行广播
        data_arr /= (max_values - min_values)[:, np.newaxis]  # 将极差沿着列方向进行广播
        labels = data[:, [0]]
        self.data = data_arr
        self.labels = labels
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        x = paddle.to_tensor([self.data[index]])
        # x.unsqueeze(1)
        label = self.labels[index]
        # print(x)
        # print(x.shape)
        if self.transform is not None:
            x = self.transform(x)
        label = label.astype('int64')
        # 返回图像和对应标签
        return x, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.labels)
