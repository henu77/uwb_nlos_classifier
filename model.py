import paddle
import paddle.nn as nn
from paddle.nn import Conv1D, MaxPool1D, AdaptiveMaxPool1D, Linear, \
    Dropout, BatchNorm1D, ReLU, Softmax, Flatten, Tanh, \
    Sigmoid
import paddle.nn.functional as F


class MyFCN_Sigmoid(nn.Layer):
    def __init__(self):
        # 调用父类的__init__函数
        super(MyFCN_Sigmoid, self).__init__()

        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            Sigmoid(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=32, out_channels=64, kernel_size=20),
            BatchNorm1D(num_features=64),
            Sigmoid(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=64, out_channels=128, kernel_size=10),
            BatchNorm1D(num_features=128),
            Sigmoid(),
            Conv1D(in_channels=128, out_channels=1, kernel_size=232),
            Sigmoid(),
            Flatten(),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyFCN_Tanh(nn.Layer):
    def __init__(self):
        # 调用父类的__init__函数
        super(MyFCN_Tanh, self).__init__()

        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            Tanh(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=32, out_channels=64, kernel_size=20),
            BatchNorm1D(num_features=64),
            Tanh(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=64, out_channels=128, kernel_size=10),
            BatchNorm1D(num_features=128),
            Tanh(),
            Conv1D(in_channels=128, out_channels=1, kernel_size=232),
            Tanh(),
            Flatten(),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyFCN_ReLU(nn.Layer):
    def __init__(self):
        # 调用父类的__init__函数
        super(MyFCN_ReLU, self).__init__()

        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            ReLU(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=32, out_channels=64, kernel_size=20),
            BatchNorm1D(num_features=64),
            ReLU(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=64, out_channels=128, kernel_size=10),
            BatchNorm1D(num_features=128),
            ReLU(),
            Conv1D(in_channels=128, out_channels=1, kernel_size=232),
            ReLU(),
            Flatten(),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyCNN_Sigmoid(nn.Layer):
    def __init__(self, dropout_rate=0):
        # 调用父类的__init__函数
        super(MyCNN_Sigmoid, self).__init__()

        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            Sigmoid(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=32, out_channels=64, kernel_size=20),
            BatchNorm1D(num_features=64),
            Sigmoid(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=64, out_channels=128, kernel_size=10),
            BatchNorm1D(num_features=128),
            Sigmoid(),
            AdaptiveMaxPool1D(output_size=20),
            Flatten(),
            Dropout(dropout_rate),
            Linear(in_features=128 * 20, out_features=64),
            Sigmoid(),
            Dropout(dropout_rate),
            Linear(in_features=64, out_features=2),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyCNN_Tanh(nn.Layer):
    def __init__(self, dropout_rate=0):
        # 调用父类的__init__函数
        super(MyCNN_Tanh, self).__init__()

        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            Tanh(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=32, out_channels=64, kernel_size=20),
            BatchNorm1D(num_features=64),
            Tanh(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=64, out_channels=128, kernel_size=10),
            BatchNorm1D(num_features=128),
            Tanh(),
            AdaptiveMaxPool1D(output_size=20),
            Flatten(),
            Dropout(dropout_rate),
            Linear(in_features=128 * 20, out_features=64),
            Tanh(),
            Dropout(dropout_rate),
            Linear(in_features=64, out_features=2),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyCNN_ReLU(nn.Layer):
    def __init__(self, dropout_rate=0):
        # 调用父类的__init__函数
        super(MyCNN_ReLU, self).__init__()

        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            ReLU(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=32, out_channels=64, kernel_size=20),
            BatchNorm1D(num_features=64),
            ReLU(),
            MaxPool1D(kernel_size=2),
            Conv1D(in_channels=64, out_channels=128, kernel_size=10),
            BatchNorm1D(num_features=128),
            ReLU(),
            AdaptiveMaxPool1D(output_size=20),
            Flatten(),
            Dropout(dropout_rate),
            Linear(in_features=128 * 20, out_features=64),
            ReLU(),
            Dropout(dropout_rate),
            Linear(in_features=64, out_features=2),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyResNet_Sigmoid(nn.Layer):
    def __init__(self, dropout_rate=0):
        # 调用父类的__init__函数
        super(MyResNet_Sigmoid, self).__init__()
        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            Sigmoid(),
            MaxPool1D(kernel_size=2, stride=2),
            MyResidual_Sigmoid(32, 64, 20, 9, 10, use_1x1conv=True),
            MaxPool1D(kernel_size=2, stride=2),
            MyResidual_Sigmoid(64, 128, 10, 4, 5, use_1x1conv=True),
            AdaptiveMaxPool1D(output_size=20),
            Flatten(),
            Dropout(dropout_rate),
            Linear(in_features=128 * 20, out_features=64),
            Sigmoid(),
            Dropout(dropout_rate),
            Linear(in_features=64, out_features=2),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyResidual_Sigmoid(nn.Layer):
    def __init__(self, input_channels, num_channels, kernel_size, padding1=1, padding2=1, use_1x1conv=False,
                 strides=1):
        super(MyResidual_Sigmoid, self).__init__()
        self.conv1 = nn.Conv1D(input_channels, num_channels, kernel_size=kernel_size,
                               padding=padding1, stride=strides)
        self.conv2 = nn.Conv1D(num_channels, num_channels, kernel_size=kernel_size,
                               padding=padding2)
        if use_1x1conv:
            self.conv3 = nn.Conv1D(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1D(num_channels)
        self.bn2 = nn.BatchNorm1D(num_channels)

    def forward(self, X):
        Y = F.sigmoid(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.sigmoid(Y)


class MyResNet_Tanh(nn.Layer):
    def __init__(self, dropout_rate=0):
        # 调用父类的__init__函数
        super(MyResNet_Tanh, self).__init__()
        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            Tanh(),
            MaxPool1D(kernel_size=2, stride=2),
            MyResidual_ReLU(32, 64, 20, 9, 10, use_1x1conv=True),
            MaxPool1D(kernel_size=2, stride=2),
            MyResidual_ReLU(64, 128, 10, 4, 5, use_1x1conv=True),
            AdaptiveMaxPool1D(output_size=20),
            Flatten(),
            Dropout(dropout_rate),
            Linear(in_features=128 * 20, out_features=64),
            Tanh(),
            Dropout(dropout_rate),
            Linear(in_features=64, out_features=2),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyResidual_Tanh(nn.Layer):
    def __init__(self, input_channels, num_channels, kernel_size, padding1=1, padding2=1, use_1x1conv=False,
                 strides=1):
        super(MyResidual_Tanh, self).__init__()
        self.conv1 = nn.Conv1D(input_channels, num_channels, kernel_size=kernel_size,
                               padding=padding1, stride=strides)
        self.conv2 = nn.Conv1D(num_channels, num_channels, kernel_size=kernel_size,
                               padding=padding2)
        if use_1x1conv:
            self.conv3 = nn.Conv1D(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1D(num_channels)
        self.bn2 = nn.BatchNorm1D(num_channels)

    def forward(self, X):
        Y = F.tanh(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.tanh(Y)


class MyResNet_ReLU(nn.Layer):
    def __init__(self, dropout_rate=0):
        # 调用父类的__init__函数
        super(MyResNet_ReLU, self).__init__()
        self.model = paddle.nn.Sequential(
            Conv1D(in_channels=1, out_channels=32, kernel_size=10),
            BatchNorm1D(num_features=32),
            ReLU(),
            MaxPool1D(kernel_size=2, stride=2),
            MyResidual_ReLU(32, 64, 20, 9, 10, use_1x1conv=True),
            MaxPool1D(kernel_size=2, stride=2),
            MyResidual_ReLU(64, 128, 10, 4, 5, use_1x1conv=True),
            AdaptiveMaxPool1D(output_size=20),
            Flatten(),
            Dropout(dropout_rate),
            Linear(in_features=128 * 20, out_features=64),
            ReLU(),
            Dropout(dropout_rate),
            Linear(in_features=64, out_features=2),
            Softmax())

    # 前向计算
    def forward(self, x):
        x = self.model(x)
        return x


class MyResidual_ReLU(nn.Layer):
    def __init__(self, input_channels, num_channels, kernel_size, padding1=1, padding2=1, use_1x1conv=False,
                 strides=1):
        super(MyResidual_ReLU, self).__init__()
        self.conv1 = nn.Conv1D(input_channels, num_channels, kernel_size=kernel_size,
                               padding=padding1, stride=strides)
        self.conv2 = nn.Conv1D(num_channels, num_channels, kernel_size=kernel_size,
                               padding=padding2)
        if use_1x1conv:
            self.conv3 = nn.Conv1D(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1D(num_channels)
        self.bn2 = nn.BatchNorm1D(num_channels)
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
