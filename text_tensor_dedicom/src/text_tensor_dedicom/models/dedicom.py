import os
import pickle
from typing import Dict, Tuple, List
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter

from text_tensor_dedicom.evaluation import evaluate
from text_tensor_dedicom.utils import get_device, set_seeds


class Dedicom(nn.Module):
    def __init__(self,
                 dim_vocab: int,
                 dim_topic: int,
                 dim_time: int,
                 M: torch.Tensor,
                 lr_a_scaling: float,
                 method=None):

        super().__init__()
        self.dim_vocab = dim_vocab
        self.dim_topic = dim_topic
        self.dim_time = dim_time

        set_seeds()
        scaling_factor = (M.mean(axis=(-1, -2)) / dim_topic ** 2) ** (1/3)
        self.A = nn.Parameter(torch.rand(1, self.dim_vocab, self.dim_topic).uniform_(0, 2) * scaling_factor.mean())
        self.R = nn.Parameter(torch.rand(self.dim_time,
                                         self.dim_topic,
                                         self.dim_topic).uniform_(0, 2) * scaling_factor.view(-1, 1, 1))
        self.U = torch.ones((1, dim_topic, dim_topic)).to(get_device())

        self.R_original = self.R.clone().to(get_device())
        self.A_original = self.A.clone().to(get_device())

        self.M = M.to(get_device())

        self.lr_a_scaling = lr_a_scaling

        self.method = method

    # row-stochastic -> linear
    @property
    def C(self):
        return self.A @ self.U

    @property
    def C_prime(self):
        return 1 / self.C

    @property
    def A_prime(self):
        return self.A / self.C

    @property
    def D(self):
        return self.A_prime / (self.C + 1e-8)

    @property
    def Q(self):
        return self.M - self.A_prime @ self.R @ self.A_prime.transpose(-1, -2)

    @property
    def X(self):
        return self.M @ self.A_prime @ self.R.transpose(-1, -2) + self.M.transpose(-1, -2) @ self.A_prime @ self.R

    @property
    def Y(self):
        return (self.A_prime @ self.R @ self.A_prime.transpose(-1, -2) @ self.A_prime @ self.R.transpose(-1, -2) +
                self.A_prime @ self.R.transpose(-1, -2) @ self.A_prime.transpose(-1, -2) @ self.A_prime @ self.R)

    @staticmethod
    def loss(target: torch.Tensor,
             reconstruction: torch.Tensor):
        reconstruction_loss_matrix = (target - reconstruction)**2
        loss = torch.sqrt(torch.sum(reconstruction_loss_matrix))
        return loss

    def update_R(self):
        with torch.no_grad():
            if self.method == 'A_rs':
                # Rowsum = 1 -> Normalization
                self.R = nn.Parameter(self.R * ((self.A_prime.transpose(-1, -2) @ self.M @ self.A_prime) /
                                                (self.A_prime.transpose(-1, -2) @ self.A_prime @ self.R @
                                                 self.A_prime.transpose(-1, -2) @ self.A_prime)))
            else:
                # non negative constraint
                multiplier = (self.A.transpose(-1, -2) @ self.M @ self.A) / (
                            self.A.transpose(-1, -2) @ self.A @ self.R @ self.A.transpose(-1, -2) @ self.A)
                alpha = self.lr_a_scaling
                self.R = nn.Parameter(self.R * (1 - alpha + alpha * multiplier))

    def update_A_frank_wolfe(self):
        with torch.no_grad():
            for t in range(100):
                A_grad = 2 * (self.A @ self.R.transpose(-1, -2) @ self.A.transpose(-1, -2) @ self.A @ self.R
                              + self.A @ self.R @ self.A.transpose(-1, -2) @ self.A @ self.R.transpose(-1, -2)
                              - self.M.transpose(-1, -2) @ self.A @ self.R
                              - self.M @ self.A @ self.R.transpose(-1, -2)).sum(dim=0)

                alpha = 2 / (t + 2)

                min_grad_col_ids = torch.argmin(A_grad, dim=-1)
                self.A *= (1 - alpha)
                self.A[:, range(self.A.shape[1]), min_grad_col_ids] += alpha

    def update_A(self):
        with torch.no_grad():
            # non negative constraint
            if self.method == 'A_nn':
                multiplier = (self.M.transpose(-1, -2) @ self.A @ self.R +
                              self.M @ self.A @ self.R.transpose(-1, -2)).sum(dim=0) / \
                             (self.A @ (self.R.transpose(-1, -2) @ self.A.transpose(-1, -2) @
                                        self.A @ self.R + self.R @ self.A.transpose(-1, -2) @
                                        self.A @ self.R.transpose(-1, -2)).sum(dim=0))
                alpha = self.lr_a_scaling
                self.A = nn.Parameter(self.A * (1 - alpha + alpha * multiplier))

            # Rowsum = 1 -> Normalization
            elif self.method == 'A_rs':
                self.A = nn.Parameter(self.A * ((self.C_prime * self.X + (self.D * self.Y) @ self.U).sum(dim=0) /
                                                (self.C_prime * self.Y + (self.D * self.X) @ self.U).sum(dim=0)))

    @staticmethod
    def col_scaled_row_softmax(mat):
        mat = mat.view(mat.shape[1], mat.shape[2])
        col_means = mat.mean(dim=0)
        col_std = mat.std(dim=0)
        mat = softmax((mat - col_means) / col_std, dim=1)
        return mat.view(1, mat.shape[0], mat.shape[1])

    def forward(self):

        if self.method == 'A_rs':
            A = self.A_prime.view(self.A_prime.shape[-2], self.A_prime.shape[-1])
            R = self.R
        elif self.method in ['A_torch', 'AR_torch']:
            A = self.col_scaled_row_softmax(self.A)
            R = self.R
        else:
            A = self.A.view(self.A.shape[-2], self.A.shape[-1])
            R = self.R
        M_recon = A @ R @ A.transpose(-1, -2)

        loss = torch.sqrt(torch.sum((M_recon - self.M)**2))

        return M_recon, loss

    def get_AR(self):
        if self.method == 'A_rs':
            return (self.A_prime.view(self.A_prime.shape[-2], self.A_prime.shape[-1]).cpu().detach().numpy(),
                    self.R.cpu().detach().numpy())
        elif self.method in ['A_torch', 'AR_torch']:
            return (self.col_scaled_row_softmax(self.A).view(self.A.shape[-2], self.A.shape[-1]).cpu().detach().numpy(),
                    self.R.cpu().detach().numpy())
        else:
            return (self.A.view(self.A.shape[-2], self.A.shape[-1]).cpu().detach().numpy(),
                    self.R.cpu().detach().numpy())

    def A_change(self):
        return torch.sqrt(torch.sum((self.A - self.A_original)**2)).detach().cpu().numpy()

    def R_change(self):
        return torch.sqrt(torch.sum((self.R - self.R_original) ** 2)).detach().cpu().numpy()

    def fit(self,
            num_epoch: int,
            save_dir: str,
            optims: Dict[str, Tuple[torch.optim.Adam, int]],
            id_to_word: Dict[int, str],
            verbose: bool,
            writer: SummaryWriter,
            evaluate_every: int,
            one_topic_per_word: bool = False,
            bin_names: List[str] = None
            ) -> Dict[str, Dict[int, float]]:
        try:
            losses = json.load(open(os.path.join(save_dir, 'losses.json'), 'r'))
        except FileNotFoundError:
            losses = {}
        losses['dedicom_loss'] = {}

        for epoch in range(num_epoch):
            epoch_losses = []

            if self.method in ['AR_torch', 'A_torch']:
                for name, (optim, n_updates) in optims.items():
                    for _ in range(n_updates):

                        if self.method == 'A_torch' and name == 'r':
                            with torch.no_grad():
                                cooc_recon, loss = self.forward()
                            epoch_losses.append(loss.item())
                            self.update_R()
                        else:
                            optim.zero_grad()
                            cooc_recon, loss = self.forward()
                            epoch_losses.append(loss.item())
                            loss.backward()
                            optim.step()
            else:
                cooc_recon, loss = self.forward()
                epoch_losses.append(loss.item())
                if self.method == 'A_fw':
                    self.update_A_frank_wolfe()
                else:
                    self.update_A()
                self.update_R()

            epoch_loss = np.mean(epoch_losses)
            epoch_A_change = self.A_change()
            epoch_R_change = self.R_change()
            writer.add_scalar('loss', epoch_loss, epoch)
            writer.add_scalar('A_change', epoch_A_change, epoch)
            writer.add_scalar('R_change', epoch_R_change, epoch)
            losses['dedicom_loss'][epoch] = epoch_loss

            if verbose and epoch % evaluate_every == 0:
                print(f'  Epoch {epoch} - Loss: {epoch_loss}')
                A, R = self.get_AR()
                pickle.dump((A, R), open(os.path.join(save_dir, 'tensors.p'), 'wb'))

                evaluate(M=A,
                         R=R,
                         epoch=epoch,
                         epoch_loss=epoch_loss,
                         id_to_word=id_to_word,
                         writer=writer,
                         save_dir=save_dir,
                         one_topic_per_word=one_topic_per_word,
                         bin_names=bin_names)
        self.M = self.M.detach().cpu()
        return losses


def initialize_training(dim_vocab: int,
                        dim_topic: int,
                        dim_time: int,
                        lr_a: float,
                        lr_a_scaling: float,
                        lr_r: float,
                        a_updates_per_epoch: int,
                        r_updates_per_epoch: int,
                        method,
                        M: torch.Tensor):

    dedicom = Dedicom(dim_topic=dim_topic,
                      dim_vocab=dim_vocab,
                      dim_time=dim_time,
                      M=M,
                      method=method,
                      lr_a_scaling=lr_a_scaling).to(get_device())

    optim_a = torch.optim.Adam([pair[1] for pair in dedicom.named_parameters() if pair[0] == 'A'], lr=lr_a)
    optim_r = torch.optim.Adam([pair[1] for pair in dedicom.named_parameters() if pair[0] == 'R'], lr=lr_r)
    optims = {
        'a': (optim_a, a_updates_per_epoch),
        'r': (optim_r, r_updates_per_epoch)
    }
    print(optims)

    return dedicom, optims


def run_dedicom(save_dir: str,
                M: torch.Tensor,
                dim_topic: int,
                dim_vocab: int,
                dim_time: int,
                lr_a: float,
                lr_a_scaling: float,
                lr_r: float,
                a_updates_per_epoch: int,
                r_updates_per_epoch: int,
                num_epochs: int,
                verbose: bool,
                evaluate_every: int,
                id_to_word: Dict[int, str],
                method=None,
                bin_names: List[str] = None,
                one_topic_per_word: bool = False):

    if bin_names is not None:
        bin_names = [name[:6] for name in bin_names]
    else:
        bin_names = [str(i+1) for i in range(M.shape[0])]

    dedicom, optims = initialize_training(dim_topic=dim_topic,
                                          dim_vocab=dim_vocab,
                                          dim_time=dim_time,
                                          lr_a=lr_a,
                                          lr_a_scaling=lr_a_scaling,
                                          lr_r=lr_r,
                                          method=method,
                                          a_updates_per_epoch=a_updates_per_epoch,
                                          r_updates_per_epoch=r_updates_per_epoch,
                                          M=M)

    writer = SummaryWriter(save_dir)
    losses = dedicom.fit(num_epoch=num_epochs,
                         save_dir=save_dir,
                         optims=optims,
                         verbose=verbose,
                         writer=writer,
                         evaluate_every=evaluate_every,
                         id_to_word=id_to_word,
                         one_topic_per_word=one_topic_per_word,
                         bin_names=bin_names)

    A, R = dedicom.get_AR()
    pickle.dump((A, R), open(os.path.join(save_dir, 'dedicom_AR.p'), 'wb'))
    json.dump(losses, open(os.path.join(save_dir, 'losses.json'), 'w'))
    return A, R, losses, writer
