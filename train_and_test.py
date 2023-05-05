import matplotlib.pyplot as plt
import numpy as np
import paddle
from tqdm import tqdm


def train(net, optimizer, epochs, batch_size, train_loader, val_loader, loss_function, save_path):
    # 设置优化器
    optim = optimizer
    # 设置损失函数
    loss_fn = loss_function
    train_acc_list = []
    train_loss_list = []
    train_predicts_list = []
    train_recall_list = []
    val_acc_list = []
    val_loss_list = []
    val_predicts_list = []
    val_recall_list = []

    now_acc = 0

    for epoch in range(epochs):
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        train_loss = 0
        train_acc = 0
        train_predicts = 0
        train_recall = 0
        train_nums = 0
        train_loop = tqdm(train_loader, desc=f"train epoch:{epoch + 1}/{epochs}")
        for idx, data in enumerate(train_loop):
            x_data = data[0]  # 训练数据
            y_data = data[1]  # 训练数据标签
            predicts = net(x_data)  # 预测结果

            # 计算损失 等价于 prepare 中loss的设置
            loss = loss_fn(predicts, y_data)

            predicts_label = predicts.argmax(axis=1)
            # 计算混淆矩阵
            tp, fn, fp, tn = get_confusion_matrix(y_data, predicts_label)
            TP += tp
            FN += fn
            FP += fp
            TN += tn
            train_loss += loss.numpy().item()
            if (TP + TN) == 0:
                train_acc = 0
            else:
                train_acc = (TP + TN) / (TP + TN + FP + FN)
            if TP == 0:
                train_predicts = 0
                train_recall = 0
            else:
                train_predicts = TP / (TP + FP)
                train_recall = TP / (TP + FN)
            # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中
            # 反向传播
            loss.backward()
            # 更新参数
            optim.step()
            # 梯度清零
            optim.clear_grad()
            train_nums += 1
            train_loop.set_postfix(
                {'loss': train_loss / train_nums, 'acc': train_acc, 'predicts': train_predicts, 'recall': train_recall})
        # 保存训练信息
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss / train_nums)
        train_predicts_list.append(train_predicts)
        train_recall_list.append(train_recall)

        val_loss = 0
        val_acc = 0
        val_predicts = 0
        val_recall = 0

        TP = 0
        FN = 0
        FP = 0
        TN = 0
        val_loop = tqdm(val_loader, desc='')
        # 禁用动态图梯度计算
        net.eval()
        val_nums = 0
        for idx, data in enumerate(val_loop):
            x_data = data[0]  # 测试数据
            y_data = data[1]  # 测试数据标签
            predicts = net(x_data)  # 预测结果

            # 计算损失与精度
            loss = loss_fn(predicts, y_data)

            predicts_label = predicts.argmax(axis=1)
            # 计算混淆矩阵
            tp, fn, fp, tn = get_confusion_matrix(y_data, predicts_label)
            TP += tp
            FN += fn
            FP += fp
            TN += tn

            val_loss += loss.numpy().item()

            if (TP + TN) == 0:
                val_acc = 0
            else:
                val_acc = (TP + TN) / (TP + TN + FP + FN)
            if TP == 0:
                val_predicts = 0
                train_recall = 0
            else:
                val_predicts = TP / (TP + FP)
                val_recall = TP / (TP + FN)
            val_nums += 1
            val_loop.set_postfix(
                {'loss': val_loss / val_nums, 'acc': val_acc, 'predicts': val_predicts, 'recall': val_recall})
        val_loss_list.append(val_loss / val_nums)
        val_acc_list.append(val_acc)
        val_predicts_list.append(val_predicts)
        val_recall_list.append(val_recall)
        # 保存模型
        if now_acc < val_acc:
            now_acc = val_acc
            paddle.save(net.state_dict(),
                        save_path + "model/" + "acc_" + str(now_acc) + "_epoch_" + str(epoch + 1) + ".pdparams")
    plt.title("acc")  # 设置图形标题
    plt.plot(range(len(train_acc_list)), train_acc_list, color='blue', linewidth=2.0,
             label='train_acc')  # 绘制曲线，设置颜色、线宽和线型
    plt.plot(range(len(val_acc_list)), val_acc_list, color='red', linewidth=2.0, label='val_acc')  # 绘制曲线，设置颜色、线宽和线型
    plt.legend(loc=0, frameon=True, facecolor='white')
    plt.savefig(save_path + "acc.png")
    plt.show()  # 显示图像

    plt.title("loss")  # 设置图形标题
    plt.plot(range(len(train_loss_list)), train_loss_list, color='blue', linewidth=2.0,
             label='train_loss')  # 绘制曲线，设置颜色、线宽和线型
    plt.plot(range(len(val_loss_list)), val_loss_list, color='red', linewidth=2.0, label='val_loss')  # 绘制曲线，设置颜色、线宽和线型
    plt.legend(loc=0, frameon=True, facecolor='white')
    plt.savefig(save_path + "loss.png")
    plt.show()  # 显示图像

    plt.title("predicts")  # 设置图形标题
    plt.plot(range(len(train_predicts_list)), train_predicts_list, color='blue', linewidth=2.0,
             label='train_predicts')  # 绘制曲线，设置颜色、线宽和线型
    plt.plot(range(len(val_predicts_list)), val_predicts_list, color='red', linewidth=2.0, label='val_predicts')
    plt.legend(loc=0, frameon=True, facecolor='white')
    plt.savefig(save_path + "predicts.png")
    plt.show()  # 显示图像

    plt.title("recall")  # 设置图形标题
    plt.plot(range(len(train_recall_list)), train_recall_list, color='blue', linewidth=2.0,
             label='train_recall')  # 绘制曲线，设置颜色、线宽和线型
    plt.plot(range(len(val_recall_list)), val_recall_list, color='red', linewidth=2.0,
             label='val_recall')  # 绘制曲线，设置颜色、线宽和线型
    plt.legend(loc=0, frameon=True, facecolor='white')
    plt.savefig(save_path + "recall.png")
    plt.show()  # 显示图像

    # 保存训练信息
    np.savetxt(save_path + 'train_acc.txt', train_acc_list)
    np.savetxt(save_path + 'train_loss.txt', train_loss_list)
    np.savetxt(save_path + 'val_acc.txt', val_acc_list)
    np.savetxt(save_path + 'val_loss.txt', val_loss_list)
    np.savetxt(save_path + 'train_predicts.txt', train_predicts_list)
    np.savetxt(save_path + 'train_recall.txt', train_recall_list)
    np.savetxt(save_path + 'val_predicts.txt', val_predicts_list)
    np.savetxt(save_path + 'val_recall.txt', val_recall_list)


def get_confusion_matrix(true_labels, pred_labels, num_classes=2):
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for true, pred in zip(true_labels, pred_labels):
        cm[true][pred] += 1
    return cm[0][0], cm[0][1], cm[1][0], cm[1][1]


def test(dataloader, model):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    test_acc = 0
    test_predicts = 0
    test_recall = 0
    for batch_id, data in enumerate(dataloader):
        x_data = data[0]  # 测试数据
        y_data = data[1]  # 测试数据标签
        predicts = model(x_data)  # 预测结果
        predicts_label = predicts.argmax(axis=1)
        # 计算混淆矩阵
        tp, fn, fp, tn = get_confusion_matrix(y_data, predicts_label)
        TP += tp
        FN += fn
        FP += fp
        TN += tn
        if (TP + TN) == 0:
            test_acc = 0
        else:
            test_acc = (TP + TN) / (TP + TN + FP + FN)
        if TP == 0:
            test_predicts = 0
            test_recall = 0
        else:
            test_predicts = TP / (TP + FP)
            test_recall = TP / (TP + FN)
    return test_acc, test_predicts, test_recall, 2 * test_predicts * test_recall / (test_predicts + test_recall)
