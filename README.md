## 一、环境配置

|   实验环境   |        版本        |
| :----------: | :----------------: |
|   操作系统   | Ubuntu 16.04.6 LTS |
|     GPU      |     Tesla V100     |
|     IDE      |    BML Codelab     |
|    Python    |       3.9.16       |
| 深度学习框架 | PaddlePaddle 2.4.1 |

## 二、项目结构

```
.
├── README.md                               # 说明文档
├── dataset                        			
│     ├──original                           # 原始数据集
│     ├──test                               # 测试集
|     ├──train                              # 训练集
|     └──val                                # 验证集
|
├── model									
│     ├──CNN.pdarams                        # CNN模型权重文件
|     ├──FCN.pdarams                        # FCN模型权重文件
|     └──ResNet.pdarams                     # ResNet模型权重文件
|
├── activate_function_experiments.ipynb     # 激活函数实验 
├── dataset.py                              # 自定义数据集类
├── divide_dataset.py                       # 划分数据集
├── dropout_experiment.ipynb                # Dropout层实验
├── model.py                                # 模型
├── test.ipynb                              # 模型评估
└── train_and_test.py                       # 训练和测试函数
```

## 三、实验步骤

### 1. 划分数据集

运行 `divide_dataset.py` 文件后，会将 `dataset/original` 文件夹下的所有数据集全部加载，打乱后按 `8:1:1` 的比例划分训练集、验证集和测试集。

### 2. 训练模型

`train_and_test.py` 文件中提供了 `train` 函数训练模型。

```python
def train(net, optimizer, epochs, batch_size, train_loader, val_loader, loss_function, save_path)
'''
	net				模型
    optimizer		优化器
    epochs			训练轮次
    batch_size		批大小
    train_loader	训练集加载器
    val_loader		验证集加载器
    loss_function	损失函数
    save_path		模型及评价指标保存路径
'''
```

例如训练模型 `MyResNet_ReLU(dropout_rate=0.5)` 代码如下：

```python
import model
import paddle
from train_and_test import train
from dataset import MyDataset

net = model.MyResNet_ReLU(dropout_rate=0.5)

batch_size = 128

train_dataset = MyDataset('./dataset/train')
eval_dataset = MyDataset('./dataset/val')

train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=batch_size)

train(net,
      epochs=300,
      optimizer=paddle.optimizer.Adam(learning_rate=0.0000001, parameters=net.parameters()),
      batch_size=batch_size,
      train_loader=train_loader,
      val_loader=eval_loader,
      loss_function=paddle.nn.CrossEntropyLoss(),
      save_path='./myresnet_relu_128_0.0000001_300/')
```

### 3. 评估

`train_and_test.py` 文件中提供了 `test` 函数训练模型。

def train(net, optimizer, epochs, batch_size, train_loader, val_loader, loss_function, save_path)
```python
def test(dataloader, model):
'''
	dataloader	数据加载器
    model		模型
'''
```

例如测试 `FCN` 模型的代码如下：

```python
import paddle
import model
from dataset import MyDataset
from train_and_test import test

weight = paddle.load('model/FCN.pdparams')

fcn = model.MyFCN_ReLU()

fcn.set_state_dict(weight)

test_dataset = MyDataset('./dataset/test')
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64, shuffle=True)

result = test(test_loader, fcn)
print(result)

```

