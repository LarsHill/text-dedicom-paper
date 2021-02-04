from collections import Counter, defaultdict
import random
from requests.exceptions import Timeout
import time
from typing import List, Optional, Dict

import numpy as np
from scipy.sparse import csr_matrix
import wikipediaapi

from text_dedicom.preprocessing import Preprocessor
from text_dedicom.utils import set_seeds

wiki = wikipediaapi.Wikipedia('en')


def create_cooc_matrix(tokens: List[str],
                       window_size: int):

    word_counter = Counter()
    word_counter.update(tokens)

    words = [word for word, count in word_counter.most_common()]
    set_seeds()
    random.shuffle(words)

    word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    vocab_len = len(word2id)

    id_tokens = [word2id[w] for w in tokens]

    cooc_mat = defaultdict(Counter)
    for i, w in enumerate(id_tokens):
        start_i = max(i - window_size, 0)
        end_i = min(i + window_size + 1, len(id_tokens))
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

    cooc = csr_matrix((xij, (i_idx, j_idx)), shape=(vocab_len, vocab_len)).toarray()
    return cooc, id2word


def get_combined_wiki_data(wiki_articles: List[str]) -> str:
    content = ''
    for article in wiki_articles:
        for i in range(30):
            try:
                page = wiki.page(article)
                content += page.text + ' '
                break
            except Timeout:
                print('TimeoutError caught (combined).')
                time.sleep(2)
                if i == 29:
                    raise
    return content


def get_word_to_article_distribution(wiki_articles: List[str],
                                     preprocessing: List[str]) -> Dict[str, List[float]]:

    # For each title, get vocab from article
    # vocab: set of str
    title2vocab = {}
    title2counter = {}

    for article in wiki_articles:
        text = None
        for i in range(30):
            try:
                page = wiki.page(article)
                text = page.text
                break
            except Timeout:
                print('TimeoutError caught (dist).')
                time.sleep(2)
                if i == 29:
                    raise

        preprocessor = Preprocessor(pipeline=preprocessing)
        text_processed = preprocessor.process(text)

        tokens = text_processed.split()

        word_counter = Counter()
        word_counter.update(tokens)

        words = [word for word, count in word_counter.most_common()]
        set_seeds()
        random.shuffle(words)

        word2id = {w: i for i, w in enumerate(words)}
        id2word = {i: w for w, i in word2id.items()}

        title2vocab[article] = set([word for word, count in word_counter.most_common()])
        title2counter[article] = word_counter

    # all words in all articles
    total_words = set([word for vocab in title2vocab.values() for word in vocab])

    # for each word, a distribution over titles, by appearance in titles list
    word2titledistribution = {word: [0]*len(wiki_articles) for word in total_words}
    # for each word, a list of all titles it appears in. Is equivalent to word2titledistribution given titles list
    word2titles = defaultdict(list)
    # unique vocab for each title
    title2uniquevocab = {}

    for title_id, title in enumerate(wiki_articles):
        vocab = title2vocab[title]
        # update title distribution for each word in article
        # later normalize to make distribution
        for word in vocab:
            word2titledistribution[word][title_id] = title2counter[title][word]
            word2titles[word].append(title)

        # get unique vocab for the article
        vocab_other = set(
            [word for voc in [value for key, value in title2vocab.items() if not key == title] for word in voc])
        vocab_unique = vocab.difference(vocab_other)
        title2uniquevocab[title] = vocab_unique

    # normalize distributions over titles
    for word, dist in word2titledistribution.items():
        word2titledistribution[word] = [x/sum(dist) for x in dist]
    return word2titledistribution, word2titles


def calc_weights(cooc: np.array,
                 alpha: float = 0.75,
                 s_max: int = 100):
    weights = (cooc / s_max) ** alpha
    weights = np.where(weights > 1, 1, weights)
    return weights


def get_input_matrix_from_wiki_data(wiki_articles: List[str],
                                    matrix_type: str,
                                    window_size: int,
                                    preprocessing: Optional[List[str]]):

    text = get_combined_wiki_data(wiki_articles)

    _, word2titles = get_word_to_article_distribution(wiki_articles=wiki_articles,
                                                                           preprocessing=preprocessing)

    preprocessor = Preprocessor(pipeline=preprocessing)
    text_processed = preprocessor.process(text)

    tokens = text_processed.split()
    cooc, id2word = create_cooc_matrix(tokens=tokens,
                                       window_size=window_size)

    if matrix_type == 'cooc':
        output = cooc
    elif matrix_type == 'ppmi':
        total_sum = cooc.sum()
        row_sums = cooc.sum(axis=1)
        col_sums = cooc.sum(axis=0)

        pxy = cooc / total_sum
        px = np.tile((row_sums / total_sum), (cooc.shape[0], 1))
        py = np.tile((col_sums / total_sum), (cooc.shape[1], 1)).T

        pmi = np.log((pxy / (px * py)) + 1e-8)
        ppmi = np.where(pmi < 0, 0., pmi)
        output = ppmi
    else:
        raise ValueError(f'Matrix type {matrix_type} is not defined.')

    return output, id2word, _, word2titles
