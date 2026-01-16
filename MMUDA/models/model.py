import torch
import torch.nn as nn
from models.ae import AE
from models.ae import VAE  
# class Model(nn.Module):         
#     def __init__(self, params):
#         super(Model, self).__init__()
#         self.params = params
#         self.ae = AE(params)
#         self.classifier = nn.Linear(512, self.params.num_of_classes)
#         # embedding_dim=512
#         # self.proj = nn.Sequential(
#         #     nn.Linear(embedding_dim, embedding_dim),
#         #     nn.LayerNorm(embedding_dim),
#         #     nn.ReLU()
#         # )


#     def forward(self, x):
#         recon, mu = self.ae(x)
#         return self.classifier(mu), recon, mu

#     def inference(self, x):
#         mu = self.ae.encoder(x)
#         return self.classifier(mu)

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.vae = VAE(params)
        self.classifier = nn.Linear(params.latent_dim, params.num_of_classes)

    def forward(self, x, use_sample_z=False):
        recon, mu, log_var, z = self.vae(x, return_z=True)

        # 选择使用 mu 或 z（采样后）作为分类输入
        latent = z if use_sample_z else mu
        logits = self.classifier(latent)

        return logits, recon, mu, log_var

    def inference(self, x):
        _, mu, _, _ = self.vae(x, return_z=True)
        return self.classifier(mu)
    

