import os
from mne.io import read_raw_edf
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample

# 设置路径
data_dir = '/media/qiu/DataDisk1/yue/data/EEG_EOG/DG/'
output_seq_root = '/media/qiu/DataDisk1/yue/SleepDG-main/data/EDF/seq/'
output_label_root = '/media/qiu/DataDisk1/yue/SleepDG-main/data/EDF/labels/'

signal_name = ['Fp1-M2', 'E1-M2']
label2id = {
    'W': 0, 'SLEEP-S0': 0,
    'N1': 1, 'SLEEP-S1': 1,
    'N2': 2, 'SLEEP-S2': 2,
    'N3': 3, 'SLEEP-S3': 3, 'N4': 3,
    'R': 4, 'SLEEP-REM': 4,
}

psg_fnames = sorted([f for f in os.listdir(data_dir) if f.endswith('.edf')])
ann_fnames = sorted([f for f in os.listdir(data_dir) if f.endswith('2.txt')])
psg_label_pairs = list(zip(psg_fnames, ann_fnames))

# 检查前两个字符是否匹配
for edf, txt in psg_label_pairs:
    edf_prefix = edf[:2]
    txt_prefix = txt[:2]
    if edf_prefix != txt_prefix:
        print(f"❌ 不匹配: {edf} 和 {txt}")
    else:
        print(f"✅ 匹配: {edf} 和 {txt}")

#
global_file_index = 0     # 子文件全局编号
folder_index = 1          # 文件夹编号

for psg_file, ann_file in tqdm(psg_label_pairs):
    raw = read_raw_edf(os.path.join(data_dir, psg_file), preload=True)

    if not all(ch in raw.ch_names for ch in signal_name):
        print(f"Skipping {psg_file} due to missing channel(s)")
        continue

    raw.pick_channels(signal_name)
    raw.filter(0.3, 35., fir_design='firwin')

    # 读取注释
    ann_path = os.path.join(data_dir, ann_file)
    with open(ann_path, 'r', errors='ignore', encoding='GBK') as f:
        lines = f.readlines()[18:]

    labels = []
    for line in lines:
        stage = line.split()[0]
        if stage in label2id:
            labels.append(label2id[stage])

    sfreq = raw.info['sfreq']
    raw_array = raw.get_data()
    epoch_samples = int(30 * sfreq)
    total_epochs = len(labels)

    X = []
    for i in range(total_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        if end > raw_array.shape[1]:
            break
        X.append(raw_array[:, start:end])
    X = np.array(X, dtype=np.float32)
    y = np.array(labels[:len(X)], dtype=np.int32)

    w_edge_epochs = 60
    sleep_idx = np.where(y != 0)[0]
    if len(sleep_idx) == 0:
        continue
    start = max(sleep_idx[0] - w_edge_epochs, 0)
    end = min(sleep_idx[-1] + w_edge_epochs + 1, len(y))
    X = X[start:end]
    y = y[start:end]

    # 标准化
    X_flat = X.transpose(0, 2, 1).reshape(-1, len(signal_name))
    X_flat = StandardScaler().fit_transform(X_flat)
    X = X_flat.reshape(-1, 30 * int(sfreq), len(signal_name)).transpose(0, 2, 1)

    if X.shape[2] != 3000:
        X = resample(X, 3000, axis=2)

    seq_len = 20
    usable_len = (len(X) // seq_len) * seq_len
    X_seq = X[:usable_len].reshape(-1, seq_len, len(signal_name), 3000)
    y_seq = y[:usable_len].reshape(-1, seq_len)

    # 用统一格式生成文件夹名（EDF-1、EDF-2、...）
    folder_name = f"EDF-{folder_index}"
    folder_index += 1

    seq_out_dir = os.path.join(output_seq_root, folder_name)
    label_out_dir = os.path.join(output_label_root, folder_name)
    os.makedirs(seq_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    for i in range(len(X_seq)):
        filename = f"{folder_name}-{global_file_index:d}.npy"
        np.save(os.path.join(seq_out_dir, filename), X_seq[i])
        np.save(os.path.join(label_out_dir, filename), y_seq[i])
        global_file_index += 1

    print(f"✅ {folder_name} 保存完成，共 {len(X_seq)} 段，当前全局编号到 {global_file_index - 1}")
