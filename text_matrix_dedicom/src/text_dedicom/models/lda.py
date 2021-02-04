import os
import pickle

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation


def run_lda(M: np.array,
            k: int,
            save_dir: str):

    lda = LatentDirichletAllocation(n_components=k, random_state=42)
    W = lda.fit_transform(M)
    H = lda.components_
    pickle.dump((W, H), open(os.path.join(save_dir, 'lda_WH.p'), 'wb'))
    return W, H
