from collections import Counter, defaultdict
import random
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from text_tensor_dedicom.data_classes import Corpus
from text_tensor_dedicom.utils import set_seeds


class Featurizer:
    def __init__(self,
                 corpus: Corpus,
                 window_size: int = 7,
                 matrix_type: str = 'ppmi',
                 vocab_size: Optional[int] = None):
        self.corpus = corpus
        self.window_size = window_size
        self.matrix_type = matrix_type
        self.vocab_size = vocab_size

        self._id_to_word: Optional[Dict[int, str]] = None
        self._word_to_id: Optional[Dict[str, int]] = None

    @property
    def id_to_word(self) -> Dict[int, str]:
        if self._id_to_word is None:
            tokens = [token for bin_ in self.corpus for article in bin_ if article.split == 'train'
                      for token in article.value_processed.split()]
            word_counter = Counter()
            word_counter.update(tokens)

            words = [word for word, count in word_counter.most_common(self.vocab_size)]
            set_seeds()
            random.shuffle(words)
            self._id_to_word = {i: w for i, w in enumerate(words)}
        return self._id_to_word

    @property
    def word_to_id(self):
        if self._word_to_id is None:
            self._word_to_id = {w: i for i, w in self.id_to_word.items()}
        return self._word_to_id

    @property
    def vocab_len(self):
        return len(self.id_to_word)

    def get_joint_articles(self) -> Dict[str, str]:
        return {bin_.id_: ' '.join([article.value_processed for article in bin_ if article.split == 'train'])
                for bin_ in self.corpus}

    @staticmethod
    def get_word_to_bins(joined_articles: Dict[str, str]) -> Dict[str, List[str]]:
        word_to_bins = defaultdict(list)
        for bin_, text in joined_articles.items():
            bin_vocab = list(set(text.split()))
            for word in bin_vocab:
                word_to_bins[word].append(bin_)

        return word_to_bins

    def assign_train_valid_test_set(self,
                                    train_split: float = 0.8,
                                    valid_split: float = 0.2):
        for bin_ in self.corpus:
            indices = list(range(len(bin_)))

            np.random.seed(42)
            np.random.shuffle(indices)

            split_train = int(len(bin_) * train_split)
            split_valid = int(len(bin_) * (train_split + valid_split))

            train_indices = indices[: split_train]
            valid_indices = indices[split_train: split_valid]
            test_indices = indices[split_valid:]

            for i in train_indices:
                bin_[i].split = 'train'
            for i in valid_indices:
                bin_[i].split = 'valid'
            for i in test_indices:
                bin_[i].split = 'test'

    @staticmethod
    def to_ppmi(cooc: np.ndarray) -> np.ndarray:
        total_sum = cooc.sum()
        row_sums = cooc.sum(axis=1)
        col_sums = cooc.sum(axis=0)

        pxy = cooc / total_sum
        px = np.tile((row_sums / total_sum), (cooc.shape[0], 1))
        py = np.tile((col_sums / total_sum), (cooc.shape[1], 1)).T

        pmi = np.log((pxy / (px * py + 1e-8)) + 1e-8)
        ppmi = np.where(pmi < 0, 0., pmi)
        return ppmi

    @staticmethod
    def to_ppmi_tensor(cooc: np.ndarray) -> np.ndarray:

        total_sum = cooc.sum()
        row_sums = cooc.sum(axis=-1).sum(axis=0)
        col_sums = cooc.sum(axis=-2).sum(axis=0)

        pxy = cooc / total_sum
        px = np.tile((row_sums / total_sum), (cooc.shape[1], 1))
        py = np.tile((col_sums / total_sum), (cooc.shape[2], 1)).T

        pmi = np.log((pxy / (px * py + 1e-8)) + 1e-8)
        ppmi = np.where(pmi < 0, 0., pmi)
        return ppmi

    @staticmethod
    def to_nppmi(cooc: np.ndarray) -> np.ndarray:
        total_sum = cooc.sum()
        row_sums = cooc.sum(axis=1)
        col_sums = cooc.sum(axis=0)

        pxy = cooc / total_sum
        px = np.tile((row_sums / total_sum), (cooc.shape[0], 1))
        py = np.tile((col_sums / total_sum), (cooc.shape[1], 1)).T

        pmi = np.log((pxy / (px * py + 1e-8)) + 1e-8)
        npmi = pmi / -np.log(pxy + 1e-8)
        nppmi = np.where(npmi < 0, 0., npmi)
        return nppmi

    @staticmethod
    def to_nppmi_tensor(cooc: np.ndarray) -> np.ndarray:

        total_sum = cooc.sum()
        row_sums = cooc.sum(axis=-1).sum(axis=0)
        col_sums = cooc.sum(axis=-2).sum(axis=0)

        pxy = cooc / total_sum
        px = np.tile((row_sums / total_sum), (cooc.shape[1], 1))
        py = np.tile((col_sums / total_sum), (cooc.shape[2], 1)).T

        pmi = np.log((pxy / (px * py + 1e-8)) + 1e-8)
        npmi = pmi / -np.log(pxy + 1e-8)
        nppmi = np.where(npmi < 0, 0., pmi)
        return nppmi

    def create_cooc_matrix(self, tokens: List[str]):

        id_tokens = [self.word_to_id[w] for w in tokens if w in self.word_to_id.keys()]

        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(id_tokens):
            start_i = max(i - self.window_size, 0)
            end_i = min(i + self.window_size + 1, len(id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j - i)

        i_idx = list()
        j_idx = list()
        xij = list()

        # Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                i_idx.append(w)
                j_idx.append(c)
                xij.append(v)

        cooc = csr_matrix((xij, (i_idx, j_idx)), shape=(self.vocab_len, self.vocab_len)).toarray()
        return cooc

    def featurize(self) -> Tuple[np.ndarray, Dict]:
        joint_articles = self.get_joint_articles()

        word_to_bins: Dict[str, List[str]] = Featurizer.get_word_to_bins(joined_articles=joint_articles)

        cooc_tensor = []
        for article in tqdm(joint_articles.values()):
            cooc = self.create_cooc_matrix(tokens=article.split())
            cooc_tensor.append(cooc)

        cooc_tensor = np.array(cooc_tensor)

        if self.matrix_type == 'cooc':
            input_tensor = cooc_tensor
        elif self.matrix_type == 'ppmi':
            input_tensor = self.to_ppmi_tensor(cooc=cooc_tensor)
        elif self.matrix_type == 'nppmi':
            input_tensor = self.to_nppmi_tensor(cooc=cooc_tensor)
        else:
            raise ValueError(f'Matrix type {self.matrix_type} is not defined.')

        return np.array(input_tensor), word_to_bins
