import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # 预定义类别频率（来自训练集统计）
        class_freq = torch.tensor([0.227, 0.115, 0.376, 0.147, 0.145], dtype=torch.float32)
        class_weights = 1.0 / class_freq 
        class_weights = class_weights / class_weights.sum()

        self.class_weights = class_weights  # 注意：device 会在 forward 中处理
        self.alpha = 0.2
        self.gamma = 2.0
        # ce 不提前构造，forward 中动态创建并迁移到正确设备

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.long()
        device = logits.device

        class_weights = self.class_weights.to(device)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)(logits, targets)

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt).pow(self.gamma)

        class_weight = class_weights[targets]
        focal_loss = -class_weight * focal_weight * torch.log(pt + 1e-8)
        focal_loss = focal_loss.mean()

        return ce_loss + self.alpha * focal_loss
