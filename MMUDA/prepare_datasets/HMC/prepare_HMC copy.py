# %%
from mne.io import read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 路径设置
dir_path = r'/media/qiu/Lenovo/yue/原数据/HMC/recordings/'
seq_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data2/HMC/seq'
label_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data2/HMC/labels'
os.makedirs(seq_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

f_names = os.listdir(dir_path)
psg_f_names = []
label_f_names = []

# 文件分类
for f_name in f_names:
    if 'sleepscoring.edf' in f_name:
        label_f_names.append(f_name)
    elif '.edf' in f_name:
        psg_f_names.append(f_name)

psg_f_names.sort()
label_f_names.sort()

# 匹配 PSG 和标签
psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:5] == label_f_name[:5]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))

# 标签映射
label2id = {
    'Sleep stage W': 0,
    'Sleep stage N1': 1,
    'Sleep stage N2': 2,
    'Sleep stage N3': 3,
    'Sleep stage R': 4,
    'Lights off@@EEG F4-A1': 0
}

# 通道设置（含 EMG chin）
signal_name = ['EEG F4-M1', 'EOG E1-M2', 'EEG C4-M1']

num_seqs = 0
num_labels = 0

# 数据处理循环
for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    epochs_list = []
    labels_list = []

    raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True)
    print(raw.info)

    try:
        raw.pick_channels(signal_name)
    except Exception as e:
        print(f"❌ 缺通道：{signal_name} in {psg_f_name}")
        continue

    raw.resample(sfreq=100)

    # 设置注释标签
    annotation = mne.read_annotations(os.path.join(dir_path, label_f_name))
    raw.set_annotations(annotation, emit_warning=False)

    events_train, event_id = mne.events_from_annotations(raw, chunk_duration=30.)
    # 移除灯光类注释
    for key in list(event_id.keys()):
        if 'Light' in key:
            event_id.pop(key)

    tmax = 30. - 1. / raw.info['sfreq']
    epochs_train = mne.Epochs(raw=raw, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)

    labels = []
    for epoch_annotation in epochs_train.get_annotations_per_epoch():
        labels.append(epoch_annotation[0][2])

    epochs = epochs_train[:]
    labels_ = labels[:]

    for epoch in epochs:
        epochs_list.append(epoch)
    for label in labels_:
        labels_list.append(label2id[label])

    index = len(epochs_list)
    while index % 20 != 0:
        index -= 1
    epochs_list = epochs_list[:index]
    labels_list = labels_list[:index]

    epochs_array_ = np.array(epochs_list)
    labels_array_ = np.array(labels_list)

    # 数据预处理 reshape + 标准化
    epochs_array_ = epochs_array_.transpose(0, 2, 1)
    epochs_array_ = epochs_array_.reshape(-1, 3)
    std = StandardScaler()
    epochs_array_ = std.fit_transform(epochs_array_)
    epochs_array_ = epochs_array_.reshape(-1, 3000, 3).transpose(0, 2, 1)

    # 序列化
    epochs_seq = epochs_array_.reshape(-1, 20, 3, 3000)
    labels_seq = labels_array_.reshape(-1, 20)

    # 保存路径构建
    if not os.path.isdir(f'{seq_dir}/{psg_f_name[:5]}'):
        os.makedirs(f'{seq_dir}/{psg_f_name[:5]}')
    for seq in epochs_seq:
        seq_name = f'{seq_dir}/{psg_f_name[:5]}/{psg_f_name[:5]}-{str(num_seqs)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    if not os.path.isdir(f'{label_dir}/{label_f_name[:5]}'):
        os.makedirs(f'{label_dir}/{label_f_name[:5]}')
    for label in labels_seq:
        label_name = f'{label_dir}/{label_f_name[:5]}/{label_f_name[:5]}-{str(num_labels)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        num_labels += 1
