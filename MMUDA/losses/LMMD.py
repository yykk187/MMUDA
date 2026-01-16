import torch
import torch.nn as nn

class mmd_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        super(mmd_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)  # [N, D]
        n_total = total.size(0)

        # 使用 torch.cdist 高效计算 pairwise 距离
        L2_distance = torch.cdist(total, total, p=2) ** 2  # [N, N]

        # 动态或固定带宽
        bandwidth = self.fix_sigma or torch.sum(L2_distance.data) / (n_total ** 2 - n_total)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        # 混合高斯核
        kernels = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernels)  # [N, N]

    def single_mmd(self, source, target):
        m, n = source.size(0), target.size(0)
        kernels = self.gaussian_kernel(source, target)

        K_ss = kernels[:m, :m]      # source-source
        K_tt = kernels[m:, m:]      # target-target
        K_st = kernels[:m, m:]      # source-target
        K_ts = kernels[m:, :m]      # target-source

        E_ss = K_ss.sum() / (m * m)
        E_tt = K_tt.sum() / (n * n)
        E_st = K_st.sum() / (m * n)
        E_ts = K_ts.sum() / (m * n)

        return E_ss + E_tt - E_st - E_ts

    def forward(self, features, domains):
        """
        features: Tensor [B, D] or [B, T, D]
        domains:  Tensor [B]
        """
        if features.dim() == 3:
            B, T, D = features.shape
            features = features.view(B * T, D)
            domains = domains.view(-1).unsqueeze(1).expand(-1, T).reshape(-1)

        unique_domains = torch.unique(domains)
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=features.device)

        domain_features = []
        for d in unique_domains:
            idx = (domains == d)
            f = features[idx]
            if f.size(0) < 2:
                domain_features.append(None)
            else:
                domain_features.append(f)

        losses = []
        for i in range(len(domain_features)):
            for j in range(i + 1, len(domain_features)):
                f1 = domain_features[i]
                f2 = domain_features[j]
                if f1 is None or f2 is None:
                    continue
                loss = self.single_mmd(f1, f2)
                losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=features.device)

        return torch.stack(losses).mean()



# ✅ 测试部分
if __name__ == '__main__':
    torch.cuda.set_device(0)
    f1 = torch.randn((64, 20, 512)).cuda()  # [B, T, D]
    f2 = f1 + 2.0  # 强制制造 domain 差异

    domain = torch.cat([torch.zeros(64), torch.ones(64)])

    features = torch.randn((128, 20, 512)).cuda()  # [B, T, D]
    domains = torch.randint(low=0, high=4, size=(128,), device='cuda:0')


    mmd_module = mmd_loss()
    loss = mmd_module(torch.cat([f1, f2], dim=0), domain)
    print("MMD Loss:", loss.item())
