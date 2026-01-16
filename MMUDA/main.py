import argparse
import torch
import numpy as np
import random
import os
import pandas as pd
from trainer import Trainer
import gc
from itertools import product
from datetime import datetime

def main():
    seed = 10
    cuda_id = 1
    dataset_list = ['SLEEPEDF', 'HMC', 'ISRUC', 'DCSM','P2018'] 

    if cuda_id >= torch.cuda.device_count():
        raise ValueError(f"❌ 无效GPU编号。你只有 {torch.cuda.device_count()} 张GPU,但你试图使用 cuda:{cuda_id}")

    torch.cuda.set_device(cuda_id)
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(cuda_id)} (cuda:{cuda_id})")
    setup_seed(seed)

    # 超参数搜索空间
    lmmd_weights = [0.01]
    contrastive_weights = [0.1]
    ae_weights = [0.01]
    dropouts = [0.1]
    lrs = [5e-3]

    # ✅ 统一结果 CSV 文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results_all_datasets_runs_{timestamp}.csv"
    write_header = True

    all_results_across_datasets = []

    for dataset in dataset_list:
        print(f"\n📂 开始处理数据集: {dataset}")
        dataset_results = []

        for run_id in range(2):  # 可设置为多轮 run
            print(f"\n🚀 开始第 {run_id + 1} 轮调参 for {dataset}")

            for lmmd_weight, contrastive_weight, ae_weight, dropout, lr in product(
                lmmd_weights, contrastive_weights, ae_weights, dropouts, lrs
            ):
                print(f"\n🔍 Run {run_id + 1} | lmmd={lmmd_weight}, contrastive={contrastive_weight}, ae_weight={ae_weight}, dropout={dropout}, lr={lr}")

                # 构造参数
                params = argparse.Namespace(
                    target_domains=[dataset],
                    cuda=cuda_id,
                    epochs=70,
                    batch_size=64,
                    num_of_classes=5,
                    lr=lr,
                    clip_value=1,
                    dropout=dropout,
                    loss_function='CrossEntropyLoss',
                    datasets_dir='data',
                    model_dir='modelsEDF',
                    num_workers=16,
                    label_smoothing=0.1,
                    latent_dim=512,
                    encoder_output_dim=512,
                    mmd_weight=lmmd_weight,
                    contrastive_weight=contrastive_weight,
                    ae_weight=ae_weight
                )

                trainer = Trainer(params, device)
                test_acc, test_f1 = trainer.train()

                result_entry = {
                    'dataset': dataset,
                    'run': run_id + 1,
                    'lmmd_weight': lmmd_weight,
                    'contrastive_weight': contrastive_weight,
                    'ae_weight': ae_weight,
                    'dropout': dropout,
                    'lr': lr,
                    'acc': test_acc,
                    'f1': test_f1
                }

                dataset_results.append(result_entry)
                all_results_across_datasets.append(result_entry)

                # ✅ 实时追加到统一 CSV 文件
                df_row = pd.DataFrame([result_entry])
                df_row.to_csv(csv_path, mode='a', index=False, header=write_header)
                write_header = False

                torch.cuda.empty_cache()
                gc.collect()

        print(f"\n📄 {dataset} 的结果已追加保存到：{csv_path}")

        # 分析每个数据集的最佳组合
        df_dataset = pd.DataFrame(dataset_results)
        best_group = (
            df_dataset.groupby(['lmmd_weight', 'contrastive_weight', 'ae_weight', 'dropout', 'lr'])[['acc', 'f1']]
            .mean()
            .sort_values('f1', ascending=False)
            .reset_index()
        )

        print(f"\n🏆 {dataset} 最佳超参数组合（按平均 F1 降序）:")
        print(best_group.head(5))

    # ✅ 所有数据集的结果分析（全局最佳）
    df_all = pd.DataFrame(all_results_across_datasets)
    print("\n📊 所有数据集整体最佳超参数组合（按平均 F1 排序）:")
    global_best = (
        df_all.groupby(['lmmd_weight', 'contrastive_weight', 'ae_weight', 'dropout', 'lr'])[['acc', 'f1']]
        .mean()
        .sort_values('f1', ascending=False)
        .reset_index()
    )
    print(global_best.head(5))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    main()
