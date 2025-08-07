import torch
from torch import nn
from torch.nn.utils import weight_norm

def causal_pad(x, pad):
    # x: (b, c, t), pad left only to enforce causality
    return nn.functional.pad(x, (pad, 0))

class residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, k, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, k, dilation=dilation))
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pad = pad

    def forward(self, x):
        # x: (b, c, t)
        y = self.relu(self.conv1(causal_pad(x, self.pad)))
        y = self.dropout(y)
        y = self.relu(self.conv2(causal_pad(y, self.pad)))
        y = self.dropout(y)
        return self.relu(y + self.downsample(x))

class tcn_backbone(nn.Module):
    def __init__(self, in_ch, hid=64, n_blocks=5, k=4, dropout=0.1):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            d = 2 ** i
            blocks.append(residual_block(in_ch if i == 0 else hid, hid, k=k, dilation=d, dropout=dropout))
        self.net = nn.Sequential(*blocks)

    @staticmethod
    def receptive_field(k, n_blocks):
        # 1 + (k-1) * sum_{i=0}^{n_blocks-1} 2^i
        return 1 + (k - 1) * (2 ** n_blocks - 1)

    def forward(self, x_seq):
        # x_seq: (b, t, d) → transpose to (b, d, t)
        h = self.net(x_seq.transpose(1, 2))          # (b, hid, t)
        return h.transpose(1, 2)                      # (b, t, hid)

class tcn_regressor(nn.Module):
    """Per-city independent regressor: last time step → scalar."""
    def __init__(self, in_ch, hid=64, n_blocks=5, k=4, dropout=0.1):
        super().__init__()
        self.backbone = tcn_backbone(in_ch, hid, n_blocks, k, dropout)
        self.head = nn.Linear(hid, 1)

    def forward(self, x_seq):
        h = self.backbone(x_seq)                      # (b, t, hid)
        return self.head(h[:, -1, :]).squeeze(-1)     # (b,)

class shared_tcn_multicity(nn.Module):
    """
    Shared backbone + city embedding + city-specific heads.
    x_seq: (b, t, d), city_ids: (b,) long
    """
    def __init__(self, in_ch, n_cities, city_embed_dim=8, hid=64, n_blocks=5, k=4, dropout=0.1):
        super().__init__()
        self.city_embed = nn.Embedding(num_embeddings=n_cities, embedding_dim=city_embed_dim)
        self.backbone   = tcn_backbone(in_ch + city_embed_dim, hid, n_blocks, k, dropout)
        self.heads      = nn.ModuleList([nn.Linear(hid, 1) for _ in range(n_cities)])

    def forward(self, x_seq, city_ids):
        # tile city embedding across time and concat to features
        emb = self.city_embed(city_ids)                          # (b, e)
        emb = emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)     # (b, t, e)
        x  = torch.cat([x_seq, emb], dim=-1)                     # (b, t, d+e)
        h  = self.backbone(x)                                    # (b, t, hid)
        out = torch.stack([head(h[i, -1, :]) for i, head in enumerate(self.heads[city_ids])])
        return out.squeeze(-1)
