import os
import pickle
from typing import Dict, Tuple
import json

import numpy as np
import torch
import torch.nn as nn

from text_tensor_dedicom.utils import get_device, set_seeds


class TNMF(nn.Module):
    def __init__(self,
                 dim_vocab: int,
                 dim_topic: int,
                 dim_time: int,
                 M: torch.Tensor,
                 device):

        super().__init__()
        self.dim_vocab = dim_vocab
        self.dim_topic = dim_topic
        self.dim_time = dim_time

        set_seeds()
        scaling_factor = (M.mean(axis=(-1, -2)) / dim_topic) ** (1 / 2)
        self.H = nn.Parameter(torch.rand(1,
                                         self.dim_topic,
                                         self.dim_vocab).uniform_(0, 2) * scaling_factor.mean(),
                              requires_grad=False).type(M.type()).to(device)
        self.W = nn.Parameter(torch.rand(self.dim_time,
                                         self.dim_vocab,
                                         self.dim_topic).uniform_(0, 2) * scaling_factor.view(-1, 1, 1),
                              requires_grad=False).type(M.type()).to(device)
        self.H = self.H * torch.sqrt(M.mean() / self.dim_topic)
        self.W = self.W * torch.sqrt(M.mean() / self.dim_topic)

        self.V = M
        self.V = self.V.to(device)
        self.V.requires_grad = False

    def loss(self,
             target: torch.Tensor,
             reconstruction: torch.Tensor):
        reconstruction_loss_matrix = (target - reconstruction) ** 2
        loss = torch.sqrt(torch.sum(reconstruction_loss_matrix))
        return loss

    def update_H(self):
        mult_H = (self.W.transpose(-1, -2) @ self.V).sum(dim=0) / ((self.W.transpose(-1, -2) @ self.W @ self.H).sum(dim=0) + 1e-10)
        self.H = self.H * mult_H

    def update_W(self):
        mult_W = (self.V @ self.H.transpose(-1, -2)) / ((self.W @ self.H @ self.H.transpose(-1, -2)) + 1e-10)
        self.W = self.W * mult_W

    def forward(self):

        M_recon = self.W @ self.H

        loss = torch.sqrt(torch.sum((M_recon - self.V) ** 2))

        return M_recon, loss

    def get_WH(self):
        self.V = self.V.detach().cpu().numpy()
        self.W = self.W.detach().cpu().numpy()
        self.H = self.H.detach().cpu().numpy()
        return self.W, self.H

    def fit(self,
            num_epoch: int,
            save_dir: str,
            evaluate_every: int = 50,
            ) -> Tuple[torch.tensor, torch.tensor, Dict]:
        try:
            losses = json.load(open(os.path.join(save_dir, 'losses.json'), 'r'))
        except FileNotFoundError:
            losses = {}
        losses['tnmf_loss'] = {}

        for epoch in range(num_epoch):
            epoch_losses = []

            cooc_recon, loss = self.forward()
            epoch_losses.append(loss.item())
            self.update_H()
            self.update_W()

            epoch_loss = np.mean(epoch_losses)

            losses['tnmf_loss'][epoch] = epoch_loss

            if epoch % evaluate_every == 0:
                print(f'  Epoch {epoch}')

        torch.cuda.empty_cache()
        W, H = self.get_WH()
        torch.cuda.empty_cache()
        return W, H, losses


def run_tnmf(M: torch.Tensor,
             save_dir: str,
             dim_topic: int,
             dim_vocab: int,
             dim_time: int,
             num_epochs: int):
    with torch.no_grad():
        tnmf = TNMF(dim_topic=dim_topic,
                    dim_vocab=dim_vocab,
                    dim_time=dim_time,
                    M=M,
                    device=get_device()).to(get_device())

        W, H, losses = tnmf.fit(num_epoch=num_epochs,
                                save_dir=save_dir)
    torch.cuda.empty_cache()
    tnmf.to('cpu')
    torch.cuda.empty_cache()
    pickle.dump((W, H), open(os.path.join(save_dir, 'tnmf_WH.p'), 'wb'))
    json.dump(losses, open(os.path.join(save_dir, 'losses.json'), 'w'))
    return W, H
