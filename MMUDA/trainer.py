import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model import Model
from tqdm import tqdm
import numpy as np
from datasets.dataset import LoadDataset
from evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from losses.double_alignment import CORAL
from losses.ae_loss import VAELoss
from losses.LMMD import mmd_loss
from timeit import default_timer as timer
import os
import copy
from losses.supcon_loss import SupConLoss
from datetime import datetime
from losses.focal_loss import CrossEntropyFocalLoss
from collections import Counter

class Trainer(object):
    def __init__(self, params, device):
        self.params = params
        self.device = device

        self.data_loader, subject_id = LoadDataset(params).get_data_loader()

        self.val_eval = Evaluator(params, self.data_loader['val'], device)
        self.test_eval = Evaluator(params, self.data_loader['test'], device)

        self.best_model_states = None

        self.model = Model(params).to(self.device)
        self.ce_loss = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).to(self.device)
        self.cf_loss = CrossEntropyFocalLoss().to(self.device)

        self.coral_loss = CORAL().to(self.device)
        self.ae_loss = VAELoss().to(self.device)
        self.contrastive_loss = SupConLoss().to(self.device)

        self.mmd_loss = mmd_loss().to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.lr, 
            weight_decay=self.params.lr / 10
        )

        self.data_length = len(self.data_loader['train'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length
        )

        print(self.model)

    def train(self):
        acc_best = 0
        f1_best = 0

        # 创建目标域迭代器
        target_loader = DataLoader(
            self.data_loader['test'].dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
            collate_fn=self.data_loader['test'].dataset.collate
        )
        target_iter = iter(target_loader)

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            loss_ce_list = []
            loss_ae_list = []
            loss_cl_list = []
            loss_mmd_list = []


            for source_batch in tqdm(self.data_loader['train'], mininterval=10):
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)

                # 源域样本
                x_s, y_s, z_s = [t.to(self.device) for t in source_batch]
                # 目标域样本（不使用 y_t）
                x_t, _, z_t = [t.to(self.device) for t in target_batch]
                # 使用 Counter 统计每个域的数量
                # z_s_values = [z.item() for z in z_s]
                # counter = Counter(z_s_values)
                # # 打印结果
                # print(counter)

                self.optimizer.zero_grad()

                # Forward pass
                pred_s, recon_s, mu_s, log_var_s = self.model(x_s)
                _, recon_t, mu_t, log_var_t = self.model(x_t)

                # 源域损失
                loss_ce = self.ce_loss(pred_s.transpose(1, 2), y_s)
                loss_ae = self.ae_loss(x_s, recon_s, mu_s, log_var_s)                
                features = mu_s.reshape(-1, mu_s.shape[-1])     # [B*T, D]
                labels = y_s.reshape(-1)                        # [B*T]
                domains = z_s.unsqueeze(1).repeat(1, y_s.shape[1]).reshape(-1)  # [B*T]

                if epoch < 15:
                    # 默认 SupCon，不管 domain
                    loss_fn = SupConLoss(temperature=0.07, cross_domain_only=False)
                    loss_cl = loss_fn(features, labels)
                else:
                    # 启用跨域正对
                    loss_fn = SupConLoss(temperature=0.07, cross_domain_only=True)
                    loss_cl = loss_fn(features, labels, domains)

                # # ✅ [可选测试] 给目标域特征添加偏移，测试 MMD 是否有效
                # if epoch == 0 :  # 添加参数控制是否启用
                #     mu_t = mu_t + 0.5 * torch.randn_like(mu_t)  # 或者 mu_t = mu_t + 2.0 试试常数偏移

                perm = torch.randperm(x_t.size(0))[:x_s.size(0)]
                x_t = x_t[perm]
                z_t = z_t[perm]

                mmd_features = torch.cat([mu_s, mu_t], dim=0)
                mmd_domains = torch.cat([z_s,z_t], dim=0)
                loss_mmd = self.mmd_loss(mmd_features, mmd_domains)
                loss = loss_ce + self.params.ae_weight *loss_ae + self.params.contrastive_weight *loss_cl+self.params.mmd_weight * loss_mmd
                # loss = loss_ce 
                loss_ce_list.append(loss_ce.item())
                loss_ae_list.append(loss_ae.item())
                loss_cl_list.append(loss_cl.item())
                loss_mmd_list.append(loss_mmd.item())



                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.scheduler.step()
                losses.append(loss.item())
            # avg_cl = np.mean(loss_cl_list)
            # print(f"[Epoch {epoch}] MMD Loss avg: {avg_cl:.4f}")
            if (epoch + 1) % 10 == 0:
                avg_ce = np.mean(loss_ce_list)
                avg_ae = np.mean(loss_ae_list)
                avg_cl = np.mean(loss_cl_list)
                avg_mmd = np.mean(loss_mmd_list)
                print(f"[E{epoch}] cf: {avg_ce.item():.3f} | ae: {avg_ae.item():.3f} | cl: {avg_cl.item():.3f} | mmd: {avg_mmd.item():.3f}")

            # 验证集评估
            with torch.no_grad():
                acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1 = self.val_eval.get_accuracy(self.model)
                print(f"Epoch {epoch+1} : Loss: {np.mean(losses):.5f}, acc: {acc:.5f}, f1: {f1:.5f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.5f}, "
                    f"Time: {(timer() - start_time) / 60:.2f} mins")
                print(cm)
                print("wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                    wake_f1, n1_f1, n2_f1, n3_f1, rem_f1))

                if acc > acc_best:
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    f1_best = f1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    print(f"Epoch {best_f1_epoch}: ACC increased!! acc: {acc_best:.5f}, f1: {f1_best:.5f}")

        print(f"{best_f1_epoch} epoch achieved best acc {acc_best:.5f} and f1 {f1_best:.5f}")
        return self.test()

    def test(self):
        self.model.load_state_dict(self.best_model_states)
        self.model.eval()

        with torch.no_grad():
            print("***************************Test************************")
            test_acc, test_f1, test_cm, test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1 = \
                self.test_eval.get_accuracy(self.model)

            print("***************************Test results************************")
            print("Test Evaluation: acc: {:.5f}, f1: {:.5f}".format(test_acc, test_f1))
            print(test_cm)
            print("wake_f1: {:.5f}, n1_f1: {:.5f}, n2_f1: {:.5f}, n3_f1: {:.5f}, rem_f1: {:.5f}".format(
                test_wake_f1, test_n1_f1, test_n2_f1, test_n3_f1, test_rem_f1))

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)

            current_date = datetime.now().strftime("%Y-%m-%d")
            model_path = os.path.join(
                self.params.model_dir,
                f"{current_date}_tacc_{test_acc:.5f}_tf1_{test_f1:.5f}.pth"
            )
            torch.save(self.best_model_states, model_path)
            print("the model is saved in " + model_path)

        return test_acc, test_f1

