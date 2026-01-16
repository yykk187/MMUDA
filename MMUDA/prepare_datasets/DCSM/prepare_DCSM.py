import mne
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 自定义函数，读取 .ids 文件并手动创建 MNE 注释对象
def read_ids_to_annotations(ids_file):
    annotations = []
    with open(ids_file, 'r') as f:
        for line in f:
            start_time, duration, label = line.strip().split(',')
            start_time = float(start_time)
            duration = float(duration)
            annotations.append((start_time, duration, label))  # (start_time, duration, label)
    
    # 将注释列表转换为 MNE 的 Annotations 对象
    onset = [ann[0] for ann in annotations]
    duration = [ann[1] for ann in annotations]
    description = [ann[2] for ann in annotations]

    ann = mne.Annotations(onset=onset, duration=duration, description=description)
    return ann

# 初始化 EEG 数据文件和标签文件的列表
psg_f_names = []
label_f_names = []

# 目录路径设置
dir_path = r'/media/qiu/Lenovo/yue/原数据/DCSM/'
seq_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data/DCSM/seq'
label_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data/DCSM/labels'

f_names = os.listdir(dir_path)
# print(f_names)

# 遍历目录下的子文件夹，忽略以 ._ 开头的文件夹和文件
for folder_name in os.listdir(dir_path):
    folder_path = os.path.join(dir_path, folder_name)
    
    # 仅处理文件夹，且忽略以 ._ 开头的文件夹
    if os.path.isdir(folder_path) and not folder_name.startswith('._'):
        print(f"Processing folder: {folder_name}")
        
        # 初始化该文件夹下的文件列表
        folder_psg_files = []
        folder_label_files = []
        
        # 遍历当前文件夹内的所有文件，跳过以 ._ 开头的文件
        for file_name in os.listdir(folder_path):
            # 跳过以 ._ 开头的文件
            if file_name.startswith('._'):
                continue  # 跳过这个文件
                
            file_path = os.path.join(folder_path, file_name)
            
            # 检查文件扩展名并分类
            if file_name.endswith('.edf'):
                folder_psg_files.append(file_name)
            elif file_name.endswith('processed.ids'):
                folder_label_files.append(file_name)
        
        # 确保 EEG 和标签文件的前六个字符相同
        for psg_file in folder_psg_files:
            prefix = psg_file[:6]  # 获取前六个字符
            # 查找标签文件是否有相同前缀
            for label_file in folder_label_files:
                if label_file[:6] == prefix:
                    psg_f_names.append(os.path.join(folder_path, psg_file))
                    label_f_names.append(os.path.join(folder_path, label_file))

# 标签映射字典
label2id = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}

num_seqs = 0
num_labels = 0
signal_name = ['F4-M1', 'E1-M2']

# 遍历 EEG 数据和标签文件
for psg_f_name, label_f_name in tqdm(zip(psg_f_names, label_f_names)):
    # 跳过以 ._ 开头的文件
    if psg_f_name.startswith('._') or label_f_name.startswith('._'):
        continue  # 跳过这对文件
    
    epochs_list = []
    labels_list = []
    
    # 读取 EEG 信号数据
    raw = mne.io.read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True)
    print(raw.info)
    raw.pick_channels(signal_name)
    raw.resample(sfreq=100)  # 重新采样

    # 使用自定义函数读取 .ids 文件并转换为 MNE 注释
    annotation = read_ids_to_annotations(os.path.join(dir_path, label_f_name))
    raw.set_annotations(annotation, emit_warning=False)

    # 创建事件
    events_train, event_id = mne.events_from_annotations(raw, chunk_duration=30.)
    print(event_id)

    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    
    epochs_train = mne.Epochs(raw=raw, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)

    print(epochs_train.event_id)

    labels = []
    for epoch_annotation in epochs_train.get_annotations_per_epoch():
        labels.append(epoch_annotation[0][2])  # 获取标签

    length = len(labels)
    epochs_train.drop_bad()
    epochs = epochs_train[:]
    labels_ = labels[:]

    # 使用断言判断 epochs 和 labels_ 的长度是否一致
    assert len(epochs) == len(labels_), f"AssertionError: Length mismatch! epochs length: {len(epochs)}, labels length: {len(labels_)}"

    print("epochs and labels have the same length.")
    print(len(epochs))
    print(len(labels_))

    # 获取所有非 'Sleep stage W' 的标签索引
    a = []
    for i, label in enumerate(labels):
        if label != 'W':
            a.append(i)

    print(0, a[0], a[-1], length)
    
    # 计算裁剪范围
    # if a[0] - 6 >= 0:
    #     start = a[0] - 6
    # else:
    #     start = 0

    if a[-1] + 50 < length:
        end = a[-1] + 50
    else:
        end = length
    print('end')
    print( end)
    start=1280

    # 确保存储的数据是裁剪后的数据，而不是所有数据
    for epoch in epochs[start:end]:  # 只保存裁剪后的数据
        epochs_list.append(epoch)
    for label in labels_[start:end]:  # 只保存裁剪后的标签
        labels_list.append(label2id[label])


    index = len(epochs_list)
    while index % 20 != 0:
        index -= 1
    epochs_list = epochs_list[:index]
    labels_list = labels_list[:index]
    print(len(epochs_list), len(labels_list))

    epochs_array_ = np.array(epochs_list)
    labels_array_ = np.array(labels_list)
    print(epochs_array_.shape, labels_array_.shape)

    # 数据预处理
    epochs_array_ = epochs_array_.transpose(0, 2, 1)
    epochs_array_ = epochs_array_.reshape(-1, 2)
    std = StandardScaler()
    epochs_array_ = std.fit_transform(epochs_array_)
    print(epochs_array_.shape)

    epochs_array_ = epochs_array_.reshape(-1, 3000, 2)
    epochs_array_ = epochs_array_.transpose(0, 2, 1)
    print(epochs_array_.shape)

    # 切分为序列
    epochs_seq = epochs_array_.reshape(-1, 20, 2, 3000)
    labels_seq = labels_array_.reshape(-1, 20)
    print(epochs_seq.shape, labels_seq.shape)
    print(epochs_seq.dtype, labels_seq.dtype)

    # 存储序列数据
    psg_dir_name = psg_f_name.split('/')[-1][:6]  # 提取文件名的前6个字符，不带路径
    if not os.path.isdir(f'{seq_dir}/{psg_dir_name}'):
        os.makedirs(f'{seq_dir}/{psg_dir_name}')
    for seq in epochs_seq:
        seq_name = f'{seq_dir}/{psg_dir_name}/{psg_dir_name}-{str(num_seqs)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    # 存储标签数据
    label_dir_name = label_f_name.split('/')[-1][:6]  # 提取文件名的前6个字符，不带路径
    if not os.path.isdir(f'{label_dir}/{label_dir_name}'):
        os.makedirs(f'{label_dir}/{label_dir_name}')
    for label in labels_seq:
        label_name = f'{label_dir}/{label_dir_name}/{label_dir_name}-{str(num_labels)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        num_labels += 1

# �� 通道名: ['E1-M2', 'E2-M2', 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'CHIN', 'LAT', 'RAT', 'SNORE', 'NASAL', 'THORAX', 'ABDOMEN', 'ECG-II', 'SPO2']