"""
Policy: CNN-Transformer Encoder (pretrained, frozen, dim=128) + trainable Decoder.
"""
import sys, os
import torch
import torch.nn as nn
from torch.distributions import Categorical

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(BASE, "CNN_Transformer3"))
from model_search import TSP_net


class RescueDecoder(nn.Module):
    def __init__(self, dim=128, dim_type=32):
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
        self.device = device; self.dim = 128
        ckpt_path = os.path.join(BASE, "CNN_Transformer3", "checkpoint", "tsp100_cnn_m5.pkl")
        args = DotDict(embedding='conv', nb_neighbors=10, kernel_size=11,
                       dim_input_nodes=2, dim_emb=128, dim_ff=512,
                       nb_layers_encoder=6, nb_layers_decoder=2,
                       nb_heads=8, max_len_PE=1000, segm_len=5, batchnorm=True)
        net = TSP_net(args.embedding, args.nb_neighbors, args.kernel_size,
                      args.dim_input_nodes, args.dim_emb, args.dim_ff,
                      args.nb_layers_encoder, args.nb_layers_decoder,
                      args.nb_heads, args.max_len_PE,
                      segm_len=args.segm_len, batchnorm=args.batchnorm)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model_baseline']); net.to(device)
        self.input_emb = net.input_emb; self.encoder = net.encoder
        self.sp = net.start_placeholder
        for p in self.input_emb.parameters(): p.requires_grad = False
        for p in self.encoder.parameters(): p.requires_grad = False
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm1d): m.track_running_stats = False
        self.encoder.eval()
        self.decoder = RescueDecoder(dim=self.dim).to(device)
        self.opt = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)

    def encode(self, coords):
        x = coords.unsqueeze(0).to(self.device)
        h = self.input_emb(x)
        bsz, N, _ = h.shape
        h = torch.cat([h, self.sp.repeat(bsz, 1, 1)], dim=1)
        h_enc, _ = self.encoder(h)
        return h_enc[:, :N, :]

    def act(self, h_enc, pos_idx, type_idx, mask, eps=0.05):
        if mask.sum() == 0:
            return 0, torch.tensor(0.0, device=h_enc.device, requires_grad=True)
        nz = mask.nonzero(as_tuple=True)[0].tolist()
        q = self.decoder(h_enc[0, pos_idx, :], type_idx)
        logits = h_enc[0] @ q
        inf = torch.tensor(float('-inf'), device=logits.device)
        logits = torch.where(mask, logits, inf)
        if torch.isnan(logits).any() or torch.rand(1).item() < eps:
            nb = [i for i in nz if i != 0]
            if nb:
                a = nb[torch.randint(0, len(nb), (1,)).item()]
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


class DotDict(dict):
    def __init__(self, **kwds): self.update(kwds); self.__dict__ = self
