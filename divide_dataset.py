import os
import numpy as np
import math


def divide_dataset3(data_path, file_prefix, train_data_path, val_data_path, test_data_path, file_size):
    # 使用os模块中的makedirs函数递归创建目录
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(val_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)
    # 获取所有csv文件的文件名列表
    csv_files = [f for f in os.listdir(data_path) if f.startswith(file_prefix)]

    all_data = np.concatenate(
        [np.loadtxt(os.path.join(data_path, file_name), delimiter=',', skiprows=1) for file_name in csv_files])
    np.random.shuffle(all_data)
    # 然后按照8:1:1的比例划分到train_data_path, val_data_path, test_data_path
    train_set = all_data[:int(len(all_data) * 0.8)]
    val_set = all_data[int(len(all_data) * 0.8):int(len(all_data) * 0.9)]
    test_set = all_data[int(len(all_data) * 0.9):]
    header = np.genfromtxt(os.path.join(data_path, csv_files[0]), delimiter=',', max_rows=1, dtype=str).tolist()
    headers = ','.join(header)

    def save_dataset(dataset, output_dir, file_prefix, file_size):
        idx = 0
        for i in range(len(dataset)):
            i += 1
            if i % file_size == 0:
                np.savetxt(os.path.join(output_dir, file_prefix + str(int(i / file_size)) + '.csv'), dataset[idx:i],
                           fmt='%d',
                           header=headers, delimiter=',',
                           comments='')
                idx = i
            if (i + 1) == len(dataset) and i % file_size != 0:
                np.savetxt(os.path.join(output_dir, file_prefix + str(math.ceil(i / file_size)) + '.csv'),
                           dataset[idx:i + 1],
                           fmt='%d',
                           header=headers, delimiter=',',
                           comments='')

    save_dataset(train_set, train_data_path, 'train_dataset_part_', file_size=file_size)
    save_dataset(val_set, val_data_path, 'val_dataset_part_', file_size=file_size)
    save_dataset(test_set, test_data_path, 'test_dataset_part_', file_size=file_size)


if __name__ == '__main__':
    divide_dataset3(data_path='./dataset/original', file_prefix='uwb_dataset', train_data_path='./dataset/train',
                    val_data_path='./dataset/val', test_data_path='./dataset/test', file_size=5000)
