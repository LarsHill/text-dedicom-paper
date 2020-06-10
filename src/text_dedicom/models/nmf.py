import json
import os
import pickle

import numpy as np
from sklearn.decomposition import NMF


def run_nmf(M: np.array,
            k: int,
            save_dir: str):

    nmf = NMF(n_components=k, init='random', random_state=42, shuffle=True, solver='mu')
    W = nmf.fit_transform(M)
    H = nmf.components_
    loss = np.linalg.norm(M - W @ H, ord='fro')
    print(f'  Final Loss: {loss}')
    pickle.dump((W, H), open(os.path.join(save_dir, 'nmf_WH.p'), 'wb'))

    try:
        losses = json.load(open(os.path.join(save_dir, 'losses.json'), 'r'))
    except FileNotFoundError:
        losses = {}
    losses['nmf_loss'] = loss
    json.dump(losses, open(os.path.join(save_dir, 'losses.json'), 'w'))
    return W, H
