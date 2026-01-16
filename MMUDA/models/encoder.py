import torch
import torch.nn as nn
import math
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.05):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value):
        B = query.shape[0]
        Q = self.q_linear(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)

        out = self.fc_out(out)
        x = self.norm1(query + out)
        return self.norm2(x + self.ffn(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleCNNLSTMEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.conv1 = nn.Sequential(
            ConvBlock(1, 64, 50, 6),
            nn.MaxPool2d((8, 1), (8, 1)),
            SEBlock(64),                                # ⬅️ Channel Attention
            nn.Dropout(params.dropout),
            ConvBlock(64, 128, 8, 1),
            SEBlock(128),
            ConvBlock(128, 128, 8, 1),
            nn.MaxPool2d((4, 1), (4, 1)),
        )

        self.conv2 = nn.Sequential(
            ConvBlock(1, 64, 400, 50),
            nn.MaxPool2d((4, 1), (4, 1)),
            SEBlock(64),                                # ⬅️ Channel Attention
            nn.Dropout(params.dropout),
            ConvBlock(64, 128, 6, 1),
            SEBlock(128),
            ConvBlock(128, 128, 6, 1),
            nn.MaxPool2d((2, 1), (2, 1)),
        )

        # 自动推导卷积后维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 3000, 2)
            x1 = self.conv1(dummy_input).view(1, -1)
            x2 = self.conv2(dummy_input).view(1, -1)
            self.rnn_input_dim = x1.size(1) + x2.size(1)

        self.rnn = nn.LSTM(self.rnn_input_dim, 256, num_layers=1,
                           dropout=params.dropout, bidirectional=True, batch_first=True)

        self.mha = MultiHeadAttention(num_heads=8, embed_dim=512, dropout=params.dropout)
        # self.fc_mu = nn.Linear(512, 512)
        # self.fc_log_var = nn.Linear(512, 512)
        self.fc_mu = nn.Linear(512, 512)
        self.fc_log_var = nn.Linear(512, 512)

    def forward(self, x):
        """
        x: [B, seq_len, 2, 3000]
        output: [B, seq_len, 512]
        """
        B, seq_len, C, T = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * seq_len, 1, T, C)  # [B*seq, 1, 3000, 2]

        x1 = self.conv1(x).squeeze(-1).reshape(B, seq_len, -1)  # [B, seq_len, feat1]
        x2 = self.conv2(x).squeeze(-1).reshape(B, seq_len, -1)  # [B, seq_len, feat2]
        x_cat = torch.cat([x1, x2], dim=-1)  # [B, seq_len, rnn_input_dim]
        # print(x_cat.shape)

        x_rnn, _ = self.rnn(x_cat)            # [B, seq_len, 512]
        # print(x_rnn.shape)
        # x_attn = self.mha(x_rnn, x_rnn, x_rnn)
        
        # mu = self.fc_mu(x_attn)
        # log_var = self.fc_log_var(x_attn)
        mu = self.fc_mu(x_rnn)
        log_var = self.fc_log_var(x_rnn)
        return mu, log_var