import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)  # 确保可复现

# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07, cross_domain_only=False):
#         """
#         Args:
#             temperature: softmax temperature
#             cross_domain_only: if True, only use same-label but different-domain as positive pairs
#         """
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.eps = 1e-8
#         self.cross_domain_only = cross_domain_only

#     def forward(self, features, labels, domains=None):
#         """
#         Args:
#             features: [B, D] - embeddings (not necessarily normalized)
#             labels: [B] - scalar class labels
#             domains: [B] - optional domain IDs
#         """
#         device = features.device
#         B = features.shape[0]

#         # Step 1: normalize features
#         features = F.normalize(features, dim=1)

#         # Step 2: cosine similarity
#         similarity_matrix = torch.matmul(features, features.T) / self.temperature

#         # Step 3: remove self-contrast
#         logits_mask = ~torch.eye(B, dtype=torch.bool, device=device)
#         similarity_matrix.masked_fill_(~logits_mask, float('-inf'))

#         # Step 4: build positive mask
#         labels = labels.view(-1, 1)
#         label_mask = torch.eq(labels, labels.T).to(device)  # [B, B]

#         if self.cross_domain_only:
#             assert domains is not None, "domains must be provided for cross-domain contrastive loss"
#             domains = domains.view(-1, 1)
#             domain_mask = torch.ne(domains, domains.T).to(device)  # only different domains
#             pos_mask = label_mask & domain_mask
#         else:
#             pos_mask = label_mask

#         # Step 5: compute log-softmax
#         logsumexp = torch.logsumexp(similarity_matrix, dim=1, keepdim=True)  # [B, 1]
#         log_prob = similarity_matrix - logsumexp  # [B, B]

#         # Step 6: apply positive mask
#         pos_mask = pos_mask.masked_fill(torch.eye(B, device=device).bool(), 0)  # remove self
#         mean_log_prob_pos = (pos_mask.float() * log_prob).sum(1) / (pos_mask.sum(1) + self.eps)

#         # Step 7: average over valid anchors
#         valid = pos_mask.sum(1) > 0
#         if valid.sum() == 0:
#             return torch.tensor(0.0, device=device)
#         loss = -mean_log_prob_pos[valid].mean()
#         return loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, cross_domain_only=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.cross_domain_only = cross_domain_only

    def forward(self, features, labels, domains=None):
        device = features.device
        N = features.shape[0]

        # Normalize
        features = F.normalize(features, dim=1)

        # Cosine similarity
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask: remove self-contrast
        logits_mask = ~torch.eye(N, dtype=torch.bool, device=device)

        # Label mask: same label = positive
        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(device)

        # Cross-domain only
        if self.cross_domain_only:
            assert domains is not None, "Must provide domains when cross_domain_only=True"
            domain_mask = torch.ne(domains.unsqueeze(1), domains.unsqueeze(0)).to(device)
            pos_mask = label_mask & domain_mask & logits_mask
        else:
            pos_mask = label_mask & logits_mask

        # Log-softmax
        exp_sim = torch.exp(sim_matrix) * logits_mask.float()
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Contrastive loss
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss



