"""
Policy: small TSPFormer (dim=64) + trainable Decoder. From scratch.
"""
import sys, os
import torch
import torch.nn as nn
from torch.distributions import Categorical

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE, "tspFormer", "tspformer"))
from tspformer.transNet import Tspformer


class RescueDecoder(nn.Module):
    def __init__(self, dim=64, dim_type=16):
        super().__init__()
        self.t_emb = nn.Embedding(3, dim_type)
        self.mlp = nn.Sequential(
            nn.Linear(dim + dim_type, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, agent_emb, type_idx):
        te = self.t_emb(torch.tensor(type_idx, device=agent_emb.device))
        return self.mlp(torch.cat([agent_emb, te], dim=0))


class RescuePolicy:
    TYPE = {'drone': 0, 'heli': 1, 'ground': 2}

    def __init__(self, device='cuda'):
        D = 64; D_FF = 256; N_ENC = 4; N_DEC = 2; N_HEAD = 4
        self.device = device; self.dim = D

        self.net = Tspformer(
            dim_input_nodes=2, dim_emb=D, dim_ff=D_FF,
            nb_layers_encoder=N_ENC, nb_layers_decoder=N_DEC,
            nb_heads=N_HEAD, max_len_PE=1000, batchnorm=False
        ).to(device)

        self.decoder = RescueDecoder(dim=D).to(device)
        self.opt = torch.optim.Adam(
            list(self.net.parameters()) + list(self.decoder.parameters()), lr=1e-4)
        self.bl = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
        self.bl_opt = torch.optim.Adam(self.bl.parameters(), lr=1e-3)

    def encode(self, coords):
        x = coords.unsqueeze(0).to(self.device)
        h = self.net.input_emb(x)
        bsz, N, _ = h.shape
        h = torch.cat([h, self.net.start_placeholder.repeat(bsz, 1, 1)], dim=1)
        h = self.net.encoder(h)
        return h[:, :N, :]

    def act(self, h_enc, pos_idx, type_idx, mask, eps=0.05):
        if mask.sum() == 0:
            return 0, torch.tensor(0.0, device=h_enc.device, requires_grad=True)
        nz = mask.nonzero(as_tuple=True)[0].tolist()
        q = self.decoder(h_enc[0, pos_idx, :], type_idx)
        logits = h_enc[0] @ q
        inf = torch.tensor(float('-inf'), device=logits.device)
        logits = torch.where(mask, logits, inf)
        if torch.isnan(logits).any() or torch.rand(1).item() < eps:
            non_base = [i for i in nz if i != 0]
            if non_base:
                a = non_base[torch.randint(0, len(non_base), (1,)).item()]
                return a, torch.tensor(0.0, device=h_enc.device, requires_grad=True)
        d = Categorical(logits=logits)
        a = d.sample(); return a.item(), d.log_prob(a)

    def act_greedy(self, h_enc, pos_idx, type_idx, mask):
        if mask.sum() == 0: return 0
        q = self.decoder(h_enc[0, pos_idx, :], type_idx)
        logits = h_enc[0] @ q
        inf = torch.tensor(float('-inf'), device=logits.device)
        logits = torch.where(mask, logits, inf)
        return logits.argmax().item()

    def value(self, h_enc, pos_idx):
        return self.bl(h_enc[0, pos_idx, :])
