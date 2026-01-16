import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy import signal
import wfdb
from wfdb.processing import resample_multichan

# ======== 配置路径 ========
dir_path = r'/media/qiu/Lenovo/yue/原数据/P2018'
seq_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data2/P2018/seq'
label_dir = r'/media/qiu/DataDisk1/yue/SleepDG-main/data2/P2018/labels'
os.makedirs(seq_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
subject_dirs = sorted(os.listdir(dir_path))

psg_label_f_pairs = [
    (f'{dir_path}/{item}/{item}.mat', f'{dir_path}/{item}/{item}.arousal', f'{dir_path}/{item}/{item}.hea')
    for item in subject_dirs
]

# ======== 标签映射 ========
label2id = {
    'W': 0,
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'R': 4
}

# ======== 统计 ========
num_seqs = 0
num_labels = 0

# ======== 所需通道 ========
required_channels = ['C3-M2', 'E1-M2', 'C4-M1']

# ======== 主循环 ========
for psg_f_name, label_f_name, hea_f_name in tqdm(psg_label_f_pairs):

    # ---------- 检查通道 ----------
    try:
        _, fields_all = wfdb.rdsamp(psg_f_name[:-4])
        sig_names = fields_all['sig_name']
        if not all(ch in sig_names for ch in required_channels):
            print(f"❌ 缺失通道于 {psg_f_name}: {required_channels}")
            continue
    except Exception as e:
        print(f"❌ 读取失败 {psg_f_name}: {e}")
        continue

    # ---------- 读取信号 ----------
    signals, fields = wfdb.rdsamp(psg_f_name[:-4], channel_names=required_channels)
    ann = wfdb.rdann(label_f_name[:-8], label_f_name[-7:])

    # ---------- 重采样 ----------
    signals, ann = resample_multichan(signals, ann, 200, 100)

    # ---------- 滤波 ----------
    b, a = signal.butter(8, [0.006, 0.7], 'bandpass')
    signals = signal.filtfilt(b, a, signals, axis=0)

    # ---------- 截取对齐 ----------
    signals = signals[ann.sample[0]:, :]
    temp = signals.shape[0] % 3000
    if temp != 0:
        signals = signals[:-temp]

    epochs_num = signals.shape[0] // 3000

    # ---------- 提取标签 ----------
    ann_labels = []
    start = 0
    for i, label in enumerate(ann.aux_note):
        if label in label2id:
            if start == 0:
                start = ann.sample[i]
            ann_labels.append((ann.sample[i] - start, label))

    if len(ann_labels) == 0:
        print(f"⚠️ 无有效标签: {psg_f_name}")
        continue

    # ---------- 标准化 ----------
    std = StandardScaler()
    signals = std.fit_transform(signals)

    # ---------- 切片 & 转置 ----------
    signals = signals.reshape(-1, 3000, 3).transpose(0, 2, 1)

    # ---------- 生成标签序列 ----------
    labels = []
    for k in range(len(ann_labels) - 1):
        begin = int(ann_labels[k][0]) // 3000
        end = int(ann_labels[k + 1][0]) // 3000
        labels += [label2id[ann_labels[k][1]]] * (end - begin)
    end = int(ann_labels[-1][0]) // 3000
    labels += [label2id[ann_labels[-1][1]]] * (epochs_num - len(labels))

    labels = np.array(labels)

    # ---------- 裁剪为20的倍数 ----------
    min_len = min(signals.shape[0], labels.shape[0])
    cut_len = min_len - (min_len % 20)
    signals = signals[:cut_len]
    labels = labels[:cut_len]

    # ---------- 组装序列 ----------
    epochs_seq = signals.reshape(-1, 20, 3, 3000)
    labels_seq = labels.reshape(-1, 20)

    # ---------- 保存 ----------
    seq_save_dir = f'{seq_dir}/{psg_f_name[-13:-4]}'
    label_save_dir = f'{label_dir}/{label_f_name[-17:-8]}'
    os.makedirs(seq_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    for seq in epochs_seq:
        seq_name = f'{seq_save_dir}/{psg_f_name[-13:-4]}-{str(num_seqs)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    for label in labels_seq:
        label_name = f'{label_save_dir}/{label_f_name[-17:-8]}-{str(num_labels)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        num_labels += 1
