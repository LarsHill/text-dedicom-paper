import json
import os
import pickle

import numpy as np

from sklearn.decomposition import TruncatedSVD


def run_truncated_svd(M: np.array,
                      k: int,
                      save_dir: str):

    svd = TruncatedSVD(n_components=k, random_state=42)
    U_sigma = svd.fit_transform(M)
    V = svd.components_
    loss = np.linalg.norm(M - U_sigma @ V, ord='fro')
    print(f'  Final Loss: {loss}')
    pickle.dump((U_sigma, V), open(os.path.join(save_dir, 'svd_UV.p'), 'wb'))

    try:
        losses = json.load(open(os.path.join(save_dir, 'losses.json'), 'r'))
    except FileNotFoundError:
        losses = {}
    losses['svd_loss'] = loss
    json.dump(losses, open(os.path.join(save_dir, 'losses.json'), 'w'))
    return U_sigma, V
