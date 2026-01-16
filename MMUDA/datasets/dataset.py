import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import random
from collections import defaultdict
from utils.util import to_tensor

def to_tensor(x):
    return torch.from_numpy(np.asarray(x))


class CustomDataset(Dataset):
    def __init__(self, seqs_labels_path_pair):
        super(CustomDataset, self).__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair
        self.channel_map = {
            "EEG": 0,
            "EOG": 1,
            "EMG": 2
        }
        self.signal_types = ('EEG', 'EOG')

    def __len__(self):
        return len(self.seqs_labels_path_pair)

    def __getitem__(self, idx):
        seq_path, label_path, domain_id = self.seqs_labels_path_pair[idx]

        data = np.load(seq_path).astype(np.float32)  # [T, C, S]
        selected_channels = []
        for sig in self.signal_types:
            ch_idx = self.channel_map[sig]
            selected_channels.append(data[:, ch_idx:ch_idx+1, :])

        seq = np.concatenate(selected_channels, axis=1)  # [T, selected_C, S]
        label = np.load(label_path).astype(np.int64)     # [T]

        return seq, label, domain_id

    def collate(self, batch):
        x_seq = np.array([x[0] for x in batch])         # [B, T, C, S]
        y_label = np.array([x[1] for x in batch])       # [B, T]
        z_domain = np.array([x[2] for x in batch])      # [B]
        return to_tensor(x_seq), to_tensor(y_label).long(), torch.tensor(z_domain).long()





class BalancedDomainSampler(Sampler):
    def __init__(self, data_source, batch_size, domain_key_idx=2):
        self.data_source = data_source
        self.batch_size = batch_size
        self.domain_key_idx = domain_key_idx

        # Step 1: 按 domain 分组
        self.domain_indices = defaultdict(list)
        for idx, item in enumerate(data_source.seqs_labels_path_pair):
            domain_id = item[self.domain_key_idx]
            self.domain_indices[domain_id].append(idx)

        self.domains = list(self.domain_indices.keys())
        self.num_domains = len(self.domains)
        self.samples_per_domain = batch_size // self.num_domains
        self.extra_samples = batch_size - (self.samples_per_domain * self.num_domains)

        # Step 2: 计算 batch 数量
        self.num_batches = min(len(indices) for indices in self.domain_indices.values()) // self.samples_per_domain

        print(f"[BalancedDomainSampler] Domains: {self.domains}, batch size: {batch_size}, "
              f"samples/domain: {self.samples_per_domain}, extra: {self.extra_samples}, "
              f"num batches: {self.num_batches}")

    def __iter__(self):
        # Step 3: 打乱每个 domain 的索引
        shuffled = {
            domain_id: random.sample(indices, len(indices))
            for domain_id, indices in self.domain_indices.items()
        }

        for i in range(self.num_batches):
            batch = []
            pool_for_extra = []

            # Step 4: 从每个 domain 抽取样本
            for domain_id in self.domains:
                start = i * self.samples_per_domain
                end = start + self.samples_per_domain
                domain_batch = shuffled[domain_id][start:end]
                batch.extend(domain_batch)

                # 收集额外候选样本
                pool_for_extra.extend(
                    shuffled[domain_id][end:end + self.extra_samples]
                )

            # Step 5: 随机补余样本
            if self.extra_samples > 0 and len(pool_for_extra) >= self.extra_samples:
                batch.extend(random.sample(pool_for_extra, self.extra_samples))

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.datasets = {
            'SLEEPEDF': 0,
            'ISRUC': 1,
            'HMC': 2,
            'P2018': 3,
            'DCSM':4,
            # 'data10':5,
        }
        dynamic_start_id = 5  # 避免和已有冲突
        for idx, domain_name in enumerate(self.params.target_domains):
            if domain_name not in self.datasets:
                self.datasets[domain_name] = dynamic_start_id + idx

        self.targets_dirs = [
            f'{self.params.datasets_dir}/{item}'
            for item in self.datasets if item in self.params.target_domains
        ]
        self.source_dirs = [
            f'{self.params.datasets_dir}/{item}'
            for item in self.datasets if item not in self.params.target_domains
        ]

    def get_data_loader(self):
        source_domains = self.load_path(self.source_dirs)
        target_domains = self.load_path(self.targets_dirs)

        train_pairs, val_pairs = self.split_dataset(source_domains)

        train_set = CustomDataset(train_pairs)
        val_set = CustomDataset(val_pairs)
        test_set = CustomDataset(target_domains)

        # 使用自定义的 BalancedDomainSampler
        train_sampler = BalancedDomainSampler(train_set, batch_size=self.params.batch_size)

        data_loader = {
            'train': DataLoader(
                train_set, batch_sampler=train_sampler,
                num_workers=self.params.num_workers, collate_fn=train_set.collate
            ),
            'val': DataLoader(
                val_set, batch_size=self.params.batch_size, shuffle=False,
                num_workers=self.params.num_workers, collate_fn=val_set.collate
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                shuffle=False,
                drop_last=True,  
                num_workers=self.params.num_workers,
                collate_fn=test_set.collate
            ),
        }
        return data_loader, None

    def load_path(self, domains_dirs):
        domains = []
        for dataset in domains_dirs:
            dataset_name = os.path.basename(dataset)
            domain_id = self.datasets[dataset_name]

            seq_dirs = sorted(os.listdir(f'{dataset}/seq'))
            labels_dirs = sorted(os.listdir(f'{dataset}/labels'))

            for seq_dir, labels_dir in zip(seq_dirs, labels_dirs):
                seq_files = sorted(os.listdir(os.path.join(dataset, 'seq', seq_dir)))
                label_files = sorted(os.listdir(os.path.join(dataset, 'labels', labels_dir)))

                for seq_file, label_file in zip(seq_files, label_files):
                    seq_path = os.path.join(dataset, 'seq', seq_dir, seq_file)
                    label_path = os.path.join(dataset, 'labels', labels_dir, label_file)
                    domains.append((seq_path, label_path, domain_id))
        return domains

    def split_dataset(self, source_domains):
        random.shuffle(source_domains)
        split_idx = int(len(source_domains) * 0.8)
        train = source_domains[:split_idx]
        val = source_domains[split_idx:]
        return train, val



