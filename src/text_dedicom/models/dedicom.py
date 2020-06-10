import os
import pickle
from typing import List, Dict, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter

from text_dedicom.evaluation import evaluate
from text_dedicom.utils import get_device, set_seeds


class Dedicom(nn.Module):
    def __init__(self,
                 n: int,
                 k: int,
                 M: torch.Tensor,
                 a_constraint: Optional[str]):

        super().__init__()
        self.n = n
        self.k = k

        set_seeds()
        self.A = nn.Parameter(torch.rand(self.n, self.k).uniform_(0, 2))
        self.R = nn.Parameter(torch.rand(self.k, self.k).uniform_(0, 2) * M.mean())

        self.a_constraint = a_constraint

        self.M = M

    def loss(self,
             target: torch.Tensor,
             reconstruction: torch.Tensor):
        reconstruction_loss_matrix = (target - reconstruction)**2
        return torch.sqrt(torch.sum(reconstruction_loss_matrix))

    @staticmethod
    def col_scaled_row_softmax(mat):
        col_means = mat.mean(dim=0)
        col_std = mat.std(dim=0)
        return softmax((mat - col_means) / col_std, dim=1)

    def forward(self,
                target: torch.Tensor = None):
        if target is None:
            target = self.M

        if self.a_constraint == 'col_scaled_row_softmax':
            M_recon = self.col_scaled_row_softmax(self.A) @ self.R @ self.col_scaled_row_softmax(self.A).T
        else:
            M_recon = self.A @ self.R @ self.A.T

        loss = self.loss(target=target, reconstruction=M_recon)
        return M_recon, loss

    def get_AR(self):
        if self.a_constraint == 'col_scaled_row_softmax':
            return (self.col_scaled_row_softmax(self.A).cpu().detach().numpy(),
                    self.R.cpu().detach().numpy())
        else:
            return (self.A.cpu().detach().numpy(),
                    self.R.cpu().detach().numpy())

    def fit(self,
            num_epoch: int,
            save_dir: str,
            optims: Dict[str, torch.optim.Adam],
            id2word: Dict[int, str],
            verbose: bool,
            writer: SummaryWriter,
            evaluate_every: int,
            ) -> Dict[str, Dict[int, float]]:
        try:
            losses = json.load(open(os.path.join(save_dir, 'losses.json'), 'r'))
        except FileNotFoundError:
            losses = {}
        losses['dedicom_loss'] = {}

        for epoch in range(num_epoch):
            epoch_losses = []
            for name, optim in optims.items():
                optim.zero_grad()
                cooc_recon, loss = self.forward()
                epoch_losses.append(loss.item())
                loss.backward()
                optim.step()

            epoch_loss = np.mean(epoch_losses)
            writer.add_scalar('loss', epoch_loss, epoch)
            losses['dedicom_loss'][epoch] = epoch_loss

            if verbose and epoch % evaluate_every == 0:
                print(f'  Epoch {epoch} - Loss: {epoch_loss}')
                A, R = self.get_AR()

                evaluate(M=A,
                         R=R,
                         epoch=epoch,
                         id2word=id2word,
                         writer=writer)
        print()
        print()

        return losses


def initialize_training(k: int,
                        n: int,
                        lr_a: float,
                        lr_r: float,
                        a_constraint: Optional[str],
                        M: torch.Tensor):

    dedicom = Dedicom(n=n,
                      k=k,
                      M=M,
                      a_constraint=a_constraint).to(get_device())

    optim_a = torch.optim.Adam([pair[1] for pair in dedicom.named_parameters() if pair[0] == 'A'], lr=lr_a)
    optim_r = torch.optim.Adam([pair[1] for pair in dedicom.named_parameters() if pair[0] == 'R'], lr=lr_r)
    optims = {'a': optim_a, 'r': optim_r}

    return dedicom, optims


def run_dedicom(save_dir: str,
                M: np.array,
                k: int,
                n: int,
                lr_a: float,
                lr_r: float,
                num_epochs: int,
                verbose: bool,
                evaluate_every: int,
                id2word: Dict[int, str],
                a_constraint: Optional[str] = None):

    dedicom, optims = initialize_training(k=k,
                                          n=n,
                                          lr_a=lr_a,
                                          lr_r=lr_r,
                                          a_constraint=a_constraint,
                                          M=M)

    writer = SummaryWriter(save_dir)
    losses = dedicom.fit(num_epoch=num_epochs,
                         save_dir=save_dir,
                         optims=optims,
                         verbose=verbose,
                         writer=writer,
                         evaluate_every=evaluate_every,
                         id2word=id2word)

    A, R = dedicom.get_AR()
    pickle.dump((A, R), open(os.path.join(save_dir, 'dedicom_AR.p'), 'wb'))
    json.dump(losses, open(os.path.join(save_dir, 'losses.json'), 'w'))
    return A, R, losses
