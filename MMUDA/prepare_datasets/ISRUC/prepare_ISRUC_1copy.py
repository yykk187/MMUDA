# %%
from mne.io import read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 数据路径
dir_path = r'/media/qiu/DataDisk1/yue/data/ISRUC'
seq_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data2/ISRUC/seq'
label_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data2/ISRUC/labels'
os.makedirs(seq_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
# 构建文件名对
psg_f_names = []
label_f_names = []
for i in range(1, 101):
    numstr = str(i)
    psg_f_names.append(f'{dir_path}/{numstr}/{numstr}.edf')
    label_f_names.append(f'{dir_path}/{numstr}/{numstr}_1.txt')

# 匹配文件对
psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:-4] == label_f_name[:-6]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))
for item in psg_label_f_pairs:
    print(item)

# 标签映射
label2id = {
    '0': 0,  # W
    '1': 1,  # N1
    '2': 2,  # N2
    '3': 3,  # N3
    '5': 4   # REM
}

# 通道名：EEG、EOG、EMG
selected_channels = ['C4-M1', 'E1-M2', 'F4-M1']

# 初始化计数
n = 0
num_seqs = 0
num_labels = 0

for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    n += 1
    labels_list = []

    try:
        raw = read_raw_edf(psg_f_name, preload=True)
        raw.pick_channels(selected_channels)
    except Exception as e:
        print(f"❌ 跳过 {psg_f_name}: {e}")
        continue

    # 预处理
    raw.resample(sfreq=100)
    raw.filter(0.3, 35, fir_design='firwin')

    # 获取信号数组 shape: (n_channels, n_samples)
    psg_array = raw.get_data().T  # shape: (n_samples, 3)
    print(f"{psg_f_name} shape: {psg_array.shape}, ch_names: {raw.ch_names}")

    # 标准化
    psg_array = StandardScaler().fit_transform(psg_array)

    # 保证整除 30 秒
    i = psg_array.shape[0] % 3000
    if i > 0:
        psg_array = psg_array[:-i, :]

    # reshape 为 30秒 epoch
    psg_array = psg_array.reshape(-1, 3000, 3)

    # 裁剪为 20 的倍数
    a = psg_array.shape[0] % 20
    if a > 0:
        psg_array = psg_array[:-a, :, :]

    # 最终 shape: (n_seq, 20, 3, 3000)
    psg_array = psg_array.reshape(-1, 20, 3000, 3)
    epochs_seq = psg_array.transpose(0, 1, 3, 2)  # (n_seq, 20, 3, 3000)

    # 读取标签
    with open(label_f_name) as f:
        for line in f:
            line_str = line.strip()
            if line_str != '':
                labels_list.append(label2id[line_str])

    labels_array = np.array(labels_list)
    if a > 0:
        labels_array = labels_array[:-a]
    labels_seq = labels_array.reshape(-1, 20)

    # 保存
    group_id = f'ISRUC-group1-{str(n)}'
    seq_save_dir = f'{seq_dir}/{group_id}'
    label_save_dir = f'{label_dir}/{group_id}'
    os.makedirs(seq_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    for seq in epochs_seq:
        np.save(f'{seq_save_dir}/{group_id}-{num_seqs}.npy', seq)
        num_seqs += 1

    for label in labels_seq:
        np.save(f'{label_save_dir}/{group_id}-{num_labels}.npy', label)
        num_labels += 1
