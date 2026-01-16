# %%
from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


dir_path = r'/media/qiu/DataDisk1/yue/SeriesSleepNet-master/data/sleep-cassette/'
f_names = os.listdir(dir_path)
print(f_names)
# %%
psg_f_names = []
label_f_names = []
for f_name in f_names:
    if 'PSG' in f_name:
        psg_f_names.append(f_name)
    if 'Hypnogram' in f_name:
        label_f_names.append(f_name)

psg_f_names.sort()
label_f_names.sort()

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:6] == label_f_name[:6]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))
print(psg_label_f_pairs)

# for item in psg_label_f_pairs:
#     print(item)

# %%
# dddd = [('SC4651E0-PSG.edf', 'SC4651EP-Hypnogram.edf'), ('SC4652E0-PSG.edf', 'SC4652EG-Hypnogram.edf')]
label2id = {'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4}
print(label2id)
# %%

signal_name = ['EEG Fpz-Cz', 'EOG horizontal']

# 定义要处理的最大对数
max_pairs = 100

# 遍历前 100 对 PSG 数据和标签文件
for count, (psg_f_name, label_f_name) in enumerate(tqdm(psg_label_f_pairs)):
    if count >= max_pairs:  # 如果已经处理了 100 对数据，就停止处理
        break

    epochs_list = []
    labels_list = []

    raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True)
    raw.pick_channels(signal_name)
    raw.filter(0.3, 35, fir_design='firwin')
    annotation = mne.read_annotations(os.path.join(dir_path, label_f_name))
    raw.set_annotations(annotation, emit_warning=False)

    events_train, event_id = mne.events_from_annotations(
        raw, chunk_duration=30.)
    if 'Sleep stage ?' in event_id.keys():
        event_id.pop('Sleep stage ?')
    if 'Movement time' in event_id.keys():
        event_id.pop('Movement time')
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    epochs_train = mne.Epochs(raw=raw, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)

    labels = []
    for epoch_annotation in epochs_train.get_annotations_per_epoch():
        labels.append(epoch_annotation[0][2])
    
    length = len(labels)
    a = []
    for i, label in enumerate(labels):
        if label != 'Sleep stage W':
            a.append(i)

    print(len(a))
    print(0, a[0], a[-1], length)
    
    if a[0] - 10 >= 0:
        start = a[0] - 10
    else:
        start = 0

    if a[-1] + 10 < length:
        end = a[-1] + 10
    else:
        end = length
    print(start, end)

    epochs = epochs_train[start:end]
    labels_ = labels[start:end]

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

    epochs_array_ = epochs_array_.transpose(0, 2, 1)
    epochs_array_ = epochs_array_.reshape(-1, 2)
    std = StandardScaler()
    epochs_array_ = std.fit_transform(epochs_array_)
    
    epochs_array_ = epochs_array_.reshape(-1, 3000, 2)
    epochs_array_ = epochs_array_.transpose(0, 2, 1)

    epochs_seq = epochs_array_.reshape(-1, 20, 2, 3000)
    labels_seq = labels_array_.reshape(-1, 20)

    # 存储前100对数据的序列和标签
    with open(r'/media/qiu/DataDisk1/yue/SleepDG-main/data/SLEEPEDF1/' + psg_f_name[3:6] + 'seq.npy', 'wb') as f:
        np.save(f, epochs_seq)
    with open(r'/media/qiu/DataDisk1/yue/SleepDG-main/data/SLEEPEDF1/' + label_f_name[3:6] + 'label.npy', 'wb') as f:
        np.save(f, labels_seq)


# %%
'''
 1: EEG Fpz-Cz
  2: EEG Pz-Oz
  3: EOG horizontal
  4: Resp oro-nasal
  5: EMG submental
  6: Temp rectal
  7: Event marker
'''
