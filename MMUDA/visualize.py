import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import umap
from tqdm import tqdm
from datasets.dataset import LoadDataset
from models.model import Model
import argparse

# ===== 字体设置 =====
font1 = {'family': 'Times New Roman'}
matplotlib.rc("font", **font1)


# ===== 固定随机种子 =====
def set_seed(seed=24):
    random.seed(seed)                  # Python 随机
    np.random.seed(seed)               # NumPy 随机
    torch.manual_seed(seed)            # PyTorch CPU
    torch.cuda.manual_seed(seed)       # 当前 GPU
    torch.cuda.manual_seed_all(seed)   # 所有 GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===== 颜色和标签映射 =====
label2class = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
label2color = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', 4: 'purple'}


# ===== 可视化模块类 =====
class Visualization3D:
    def __init__(self, params, seed=24):
        self.params = params
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.data_loader, _ = LoadDataset(params).get_data_loader()
        self.model = Model(params).cuda()
        self.model.load_state_dict(torch.load(self.params.model_path, map_location='cpu'))

    def extract_features(self, max_points_per_class=1000):
        ts = umap.UMAP(n_components=3, random_state=self.seed)
        self.model.eval()
        feats_list, labels_list = [], []

        for x, y, _ in tqdm(self.data_loader['test']):
            x = x.cuda()
            y = y.cuda()
            _, _, mu, _ = self.model(x)
            feats = mu.view(-1, mu.shape[-1])
            feats_list.append(feats.detach().cpu().numpy())
            labels_list.append(y.view(-1).detach().cpu().numpy())

        feats_all = np.concatenate(feats_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)

        sampled_feats, sampled_labels = [], []
        for class_id in np.unique(labels_all):
            idx = np.where(labels_all == class_id)[0]
            if len(idx) > max_points_per_class:
                idx = self.rng.choice(idx, max_points_per_class, replace=False)
            sampled_feats.append(feats_all[idx])
            sampled_labels.append(labels_all[idx])

        feats_all = np.concatenate(sampled_feats, axis=0)
        labels_all = np.concatenate(sampled_labels, axis=0)

        feats_ts = ts.fit_transform(feats_all)
        return feats_ts, labels_all


def extract_domain_features(vis, max_points=4000, seed=24):
    ts = umap.UMAP(n_components=3, random_state=seed)
    rng = np.random.RandomState(seed)
    vis.model.eval()

    def extract(dataloader, with_domain=False):
        feats_list = []
        domain_labels = []
        for x, _, z in tqdm(dataloader):
            x = x.cuda()
            _, _, mu, _ = vis.model(x)
            feats = mu.view(-1, mu.shape[-1]).detach().cpu().numpy()
            feats_list.append(feats)
            if with_domain:
                domain_labels.extend(z.view(-1).detach().cpu().numpy())
        if with_domain:
            return np.concatenate(feats_list, axis=0), np.array(domain_labels)
        else:
            return np.concatenate(feats_list, axis=0)

    feats_source, domain_ids = extract(vis.data_loader['val'], with_domain=True)
    feats_target = extract(vis.data_loader['test'])

    feats_src_filtered = []
    domain_filtered = []
    for dom in np.unique(domain_ids):
        idx = np.where(domain_ids == dom)[0]
        if len(idx) > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        feats_src_filtered.append(feats_source[idx])
        domain_filtered.extend([dom] * len(idx))

    feats_source = np.concatenate(feats_src_filtered, axis=0)
    domain_ids = np.array(domain_filtered)

    if feats_target.shape[0] > max_points:
        feats_target = feats_target[rng.choice(feats_target.shape[0], max_points, replace=False)]

    feats_all = np.concatenate([feats_source, feats_target], axis=0)
    domain_all = np.concatenate([domain_ids, [-1] * len(feats_target)])

    feats_ts = ts.fit_transform(feats_all)
    return feats_ts, domain_all


# ===== 网格绘制函数（两个模式）=====
def draw_background_grid_planes(ax, range_x, range_y, range_z, step=1.0, color='lightgray', alpha=1, linewidth=0.5):
    for x in np.arange(*range_x, step):
        ax.plot([x, x], [range_y[0], range_y[1]], [range_z[0]]*2, color=color, alpha=alpha, linewidth=linewidth)
    for y in np.arange(*range_y, step):
        ax.plot([range_x[0], range_x[1]], [y, y], [range_z[0]]*2, color=color, alpha=alpha, linewidth=linewidth)
    for x in np.arange(*range_x, step):
        ax.plot([x, x], [range_y[0]]*2, [range_z[0], range_z[1]], color=color, alpha=alpha, linewidth=linewidth)
    for z in np.arange(*range_z, step):
        ax.plot([range_x[0], range_x[1]], [range_y[0]]*2, [z, z], color=color, alpha=alpha, linewidth=linewidth)
    for y in np.arange(*range_y, step):
        ax.plot([range_x[0]]*2, [y, y], [range_z[0], range_z[1]], color=color, alpha=alpha, linewidth=linewidth)
    for z in np.arange(*range_z, step):
        ax.plot([range_x[0]]*2, [range_y[0], range_y[1]], [z, z], color=color, alpha=alpha, linewidth=linewidth)


def draw_3d_grid(ax, range_x, range_y, range_z, step=0.2, color='lightgray', linewidth=0.5):
    for z in np.arange(*range_z, step):
        for x in np.arange(*range_x, step):
            ax.plot([x, x], [range_y[0], range_y[1]], [z, z], color=color, linewidth=linewidth)
        for y in np.arange(*range_y, step):
            ax.plot([range_x[0], range_x[1]], [y, y], [z, z], color=color, linewidth=linewidth)
    for x in np.arange(*range_x, step):
        for y in np.arange(*range_y, step):
            ax.plot([x, x], [y, y], [range_z[0], range_z[1]], color=color, linewidth=linewidth)


# ===== 绘图主函数（feature/domain 两个版本保持一致风格）=====
def draw_feature_3d(
    feats_ts,
    labels_all,
    save_path,
    grid_mode='background',
    background_shrink=0.2
):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    for class_id in np.unique(labels_all):
        idx = np.where(labels_all == class_id)[0]
        ax.scatter(
            feats_ts[idx, 0], feats_ts[idx, 1], feats_ts[idx, 2],
            c=colors[class_id],
            edgecolor=colors[class_id],
            linewidths=0.3,
            label=labels[class_id],
            s=7,
            alpha=0.4
        )

    min_x, max_x = feats_ts[:, 0].min(), feats_ts[:, 0].max()
    min_y, max_y = feats_ts[:, 1].min(), feats_ts[:, 1].max()
    min_z, max_z = feats_ts[:, 2].min(), feats_ts[:, 2].max()

    ax.set_xlim(min_x + background_shrink, max_x - background_shrink)
    ax.set_ylim(min_y + background_shrink, max_y - background_shrink)
    ax.set_zlim(min_z + background_shrink, max_z - background_shrink)

    range_x = (min_x + background_shrink, max_x - background_shrink)
    range_y = (min_y + background_shrink, max_y - background_shrink)
    range_z = (min_z + background_shrink, max_z - background_shrink)

    if grid_mode == 'full':
        draw_3d_grid(ax, range_x, range_y, range_z, step=1.0)
    elif grid_mode == 'background':
        draw_background_grid_planes(ax, range_x, range_y, range_z, step=1.0)

    ax.xaxis.pane.set_visible(True)
    ax.yaxis.pane.set_visible(True)
    ax.zaxis.pane.set_visible(True)
    ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=30)
    leg = ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.0, 0.9))
    leg.get_frame().set_alpha(0.5)   # ✅ 半透明
    


    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    print(f"✅ Saved 3D feature plot to: {save_path}")


def draw_domain_3d(
    feats_ts,
    domain_labels,
    save_path,
    grid_mode='background',
    background_shrink=0.2
):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    domain_ids = np.unique(domain_labels)
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink']

    for i, domain in enumerate(domain_ids):
        idx = np.where(domain_labels == domain)[0]
        xs, ys, zs = feats_ts[idx, 0], feats_ts[idx, 1], feats_ts[idx, 2]
        color = 'blue' if domain == -1 else colors[i % len(colors)]
        label = 'Target Domain' if domain == -1 else f"Source Domain {domain}"
        ax.scatter(
            xs, ys, zs,
            c=color,
            edgecolor=color,
            linewidths=0.3,
            label=label,
            s=7,
            alpha=0.4
        )

    min_x, max_x = feats_ts[:, 0].min(), feats_ts[:, 0].max()
    min_y, max_y = feats_ts[:, 1].min(), feats_ts[:, 1].max()
    min_z, max_z = feats_ts[:, 2].min(), feats_ts[:, 2].max()

    ax.set_xlim(min_x + background_shrink, max_x - background_shrink)
    ax.set_ylim(min_y + background_shrink, max_y - background_shrink)
    ax.set_zlim(min_z + background_shrink, max_z - background_shrink)

    range_x = (min_x + background_shrink, max_x - background_shrink)
    range_y = (min_y + background_shrink, max_y - background_shrink)
    range_z = (min_z + background_shrink, max_z - background_shrink)

    if grid_mode == 'full':
        draw_3d_grid(ax, range_x, range_y, range_z, step=1.0)
    elif grid_mode == 'background':
        draw_background_grid_planes(ax, range_x, range_y, range_z, step=1.0)

    ax.xaxis.pane.set_visible(True)
    ax.yaxis.pane.set_visible(True)
    ax.zaxis.pane.set_visible(True)
    ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
    ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=30)
    leg = ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.0, 0.9))
    leg.get_frame().set_alpha(0.5)   # ✅ 半透明

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    print(f"✅ Saved 3D domain alignment plot to: {save_path}")


# ===== 主程序入口 =====
if __name__ == '__main__':
    set_seed(42)  # ✅ 固定随机种子

    class Args:
        target_domains = ['ISRUC']
        datasets_dir = 'data'
        base_model_path = 'modelstiaocan/2025-08-31_tacc_0.54083_tf1_0.33351.pth'
        dg_model_path = 'modelstiaocan/2025-07-31_tacc_0.81340_tf1_0.76704.pth'
        cuda = 1

    args = Args()

    # ===== BASE模型可视化 =====
    print("✅ Visualizing BASE model")
    params = argparse.Namespace(
        target_domains=args.target_domains,
        datasets_dir=args.datasets_dir,
        dropout=0.1,
        latent_dim=512,
        encoder_output_dim=512,
        batch_size=64,
        num_workers=16,
        num_of_classes=5,
        model_path=args.base_model_path
    )

    vis_base = Visualization3D(params, seed=17)
    feats_ts, labels_all = vis_base.extract_features(max_points_per_class=1000)
    draw_feature_3d(feats_ts, labels_all, save_path='base_feature_EDF_3dISRUC.png', grid_mode='background')

    feats_ts, domain_labels = extract_domain_features(vis_base, max_points=1000, seed=12)
    draw_domain_3d(feats_ts, domain_labels, save_path='base_domain_EDF_3dISRUC.png', grid_mode='background')

    # ===== DG模型可视化 =====
    params.model_path = args.dg_model_path
    print("✅ Visualizing DG model")
    vis_dg = Visualization3D(params, seed=17)
    feats_ts, labels_all = vis_dg.extract_features(max_points_per_class=1000)
    draw_feature_3d(feats_ts, labels_all, save_path='dg_feature_EDF_3dISRUC.png', grid_mode='background')

    feats_ts, domain_labels = extract_domain_features(vis_dg, max_points=1000, seed=12)
    draw_domain_3d(feats_ts, domain_labels, save_path='dg_domain_EDF_3dISRUC.png', grid_mode='background')
