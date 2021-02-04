from collections import Counter
import json
import os
import pickle
from typing import List, Dict, Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from text_dedicom.visualization import heatmap_plot, losses_plot, scatterplot_colored
from text_dedicom.visualization import fig_to_tensorboard, df_to_txt, save_fig
from text_dedicom.models.dimensionality_reduction import umap_transform


def calc_cosine_similarities(embeddings_1: np.array,
                             embeddings_2: np.array) -> np.array:
    dot_products = embeddings_1.dot(embeddings_2.T)
    norm_1 = np.linalg.norm(embeddings_1, ord=2, axis=1)
    norm_2 = np.linalg.norm(embeddings_2, ord=2, axis=1)
    dotted_norms = norm_1.reshape(norm_1.shape[0], 1).dot(norm_2.reshape(norm_2.shape[0], 1).T)
    pairwise_cos_sim = np.divide(dot_products, dotted_norms)
    return pairwise_cos_sim


def get_similar_words(M: np.array,
                      n_topic_words: int,
                      n_similar_words: int,
                      id2word: Dict[int, str],
                      topics: Dict[str, List[Tuple[str, int, np.array]]]) -> Dict[str, List[str]]:
    M = M if M.shape[0] > M.shape[1] else M.T

    similar_words = {}
    for topic, words in topics.items():
        if words:
            embeddings = np.array([item[2] for item in words])[:n_topic_words, :]
            indices = np.array([item[1] for item in words])[:n_topic_words]
            cos_sims = calc_cosine_similarities(embeddings, M).astype(dtype='float64')
            # Set cos_sim of topic word with topic word to 0
            for row_ind, col_ind in enumerate(indices):
                cos_sims[row_ind, col_ind] = 0.
            cos_sims_sorted_indices = np.argsort(cos_sims, axis=1)[:, ::-1][:, :n_similar_words]
            cos_sims_sorted = np.take_along_axis(cos_sims, indices=cos_sims_sorted_indices, axis=1)

            topic_column = []
            for topic_word_ind, topic_word_cos_sim, ind in zip(cos_sims_sorted_indices, cos_sims_sorted, indices):
                topic_column.append(f'{id2word[int(ind)]}')
                topic_column.append('(1.0)')
                for idx, cos_sim in zip(topic_word_ind, topic_word_cos_sim):
                    topic_column.append(f'{id2word[int(idx)]}')
                    topic_column.append(f'({np.round(cos_sim, 3)})')
        else:
            topic_column = ['N/A'] * n_topic_words * (n_similar_words + 1) * 2

        while len(topic_column) < n_topic_words * (n_similar_words + 1) * 2:
            topic_column.append('N/A')

        similar_words[topic] = topic_column

    return similar_words


def get_topics(M: np.array,
               n: int,
               id2word: Dict[int, str],
               save_dir: str = None) -> Tuple[Dict[str, List[str]],
                                              Dict[str, List[Tuple[str, int, np.array]]]]:
    M = M if M.shape[0] > M.shape[1] else M.T

    topics_paper = {}
    topics_all = {}
    for dim in range(M.shape[1]):
        topics_paper[f'Topic {dim + 1}'] = []
        topics_all[f'Topic {dim + 1}'] = []

        embeddings = M[np.argwhere(M.argmax(axis=1) == dim).flatten()]
        indices = np.argwhere(M.argmax(axis=1) == dim).flatten()

        new_sorting_order = np.argsort(embeddings[:, dim], axis=0)[::-1]

        embeddings_sorted = embeddings[new_sorting_order].astype(dtype='float64')
        indices_sorted = indices[new_sorting_order]

        counter = 0
        for embedding, id_ in zip(embeddings_sorted, indices_sorted):

            if counter < n:
                topics_paper[f'Topic {dim + 1}'].append(id2word[id_])
                topics_paper[f'Topic {dim + 1}'].append(f'({np.round(embedding.max(), 3)})')

            if counter >= n and not save_dir:
                break
            topics_all[f'Topic {dim + 1}'].append((id2word[id_], id_, embedding))

            counter += 1

        if embeddings_sorted.shape[0] < n:
            while len(topics_paper[f'Topic {dim + 1}']) < (n * 2):
                topics_paper[f'Topic {dim + 1}'].append('N/A')

    if save_dir:
        pickle.dump(topics_all, open(os.path.join(save_dir, 'topics_all.p'), 'wb'))

    return topics_paper, topics_all


def get_topics_old(M: np.array,
                   n: int,
                   id2word: Dict[int, str]) -> Dict[int, List[str]]:
    M = M if M.shape[0] > M.shape[1] else M.T

    topics = {f'Topic {i}': [id2word[int(idx)] for idx in column]
              for i, column in enumerate(np.argsort(M, axis=0)[::-1, ][:n, ].T, 1)}

    return topics


def get_topic_counts_of_vocab(M: np.array) -> Dict:
    M = M if M.shape[0] > M.shape[1] else M.T

    topic_counts = Counter(M.argmax(axis=1))
    topic_counts = {f'Topic {topic + 1}': count for topic, count in sorted(topic_counts.items())}
    return topic_counts


def evaluate_topic_sparsity(M: np.array,
                            n: int,
                            id2word: Dict[int, str],
                            word2titledistribution: Dict[str, List[float]],
                            ord: float = 0.5) -> List[float]:
    topics, _ = get_topics(M, n, id2word)
    num_titles = list(word2titledistribution.values())[0].__len__()
    sparsities = []
    for topic in topics.values():
        dist = np.zeros(num_titles)
        for word in topic:
            try:
                dist += np.array(word2titledistribution[word])
            except KeyError:
                pass
        dist /= np.sum(dist)
        sparsities.append(np.linalg.norm(dist, ord=ord))
    return sparsities


def evaluate(M: np.array,
             id2word: Dict[int, str],
             R: Optional[np.array] = None,
             epoch: Optional[int] = None,
             writer: Optional[SummaryWriter] = None) -> Tuple[Dict, Dict]:

    topics, topics_all = get_topics(M,
                                    n=10,
                                    id2word=id2word,
                                    save_dir=None)
    similar_words = get_similar_words(M,
                                      n_topic_words=2,
                                      n_similar_words=4,
                                      topics=topics_all,
                                      id2word=id2word)

    if writer:
        heatmap = fig_to_tensorboard(heatmap_plot(R))
        topic_text = '=' * 64 + '\n\n'
        for i, (topic) in enumerate(topics.values()):
            topic_text += f'TOPIC {i}: '
            topic_text += '[' + ' '.join(topic) + ' ]\n\n'
        writer.add_text('topics', topic_text, epoch)
        writer.add_image('heatmap/R', heatmap, epoch)

    plt.close('all')
    return topics, similar_words


def generate_paper_output(save_dir: str,
                          M: np.array,
                          method: str,
                          titles: List[str],
                          word2titles: Dict[str, List[str]],
                          id2word: Dict[int, str],
                          losses: Optional[Dict] = None,
                          R: Optional[np.array] = None):

    json.dump(id2word, open(os.path.join(save_dir, 'id2word.json'), 'w'))
    json.dump(word2titles, open(os.path.join(save_dir, 'word2titles.json'), 'w'))

    topics, topics_all = get_topics(M,
                                    n=10,
                                    id2word=id2word,
                                    save_dir=save_dir)
    topic_counts = get_topic_counts_of_vocab(M)
    similar_words = get_similar_words(M,
                                      n_topic_words=2,
                                      n_similar_words=4,
                                      topics=topics_all,
                                      id2word=id2word)

    json.dump(topic_counts, open(os.path.join(save_dir, f'topic_counts_{method}.json'), 'w'))
    json.dump(topics, open(os.path.join(save_dir, f'topics_{method}.json'), 'w'))
    json.dump(similar_words, open(os.path.join(save_dir, f'similar_words_{method}.json'), 'w'))

    # Save Topic Table
    df_to_txt(df=pd.DataFrame.from_dict(topics),
              save_dir=os.path.join(save_dir, 'output'),
              save_name=f'tab_topics_{method}.txt')

    # Save Topic Count Table
    df_to_txt(df=pd.DataFrame(topic_counts, index=[0]),
              save_dir=os.path.join(save_dir, 'output'),
              save_name=f'tab_topic_counts_{method}.txt')

    # Save Similar Words Table
    df_to_txt(df=pd.DataFrame.from_dict(similar_words),
              save_dir=os.path.join(save_dir, 'output'),
              save_name=f'tab_similar_words_{method}.txt')

    # Save dimensionality reduction plots
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M_umap = umap_transform(M)

    pickle.dump(M_umap, open(os.path.join(save_dir, f'umap_{method}.p'), 'wb'))
    fig_umap_titles = scatterplot_colored(M_umap, titles=titles, word2titles=word2titles, id2word=id2word)
    save_fig(fig_umap_titles,
             save_dir=save_dir,
             save_name=f'scatterplot_titles_umap_{method}')
    plt.close(fig_umap_titles)

    fig_umap_topics = scatterplot_colored(M_umap, topics_all=topics_all, id2word=id2word)
    save_fig(fig_umap_topics,
             save_dir=save_dir,
             save_name=f'scatterplot_topics_umap_{method}')
    plt.close(fig_umap_topics)

    # Save losses/Sparsity Score Figure
    if losses:
        fig_losses = losses_plot(losses)
        save_fig(fig_losses,
                 save_dir=save_dir,
                 save_name=f'losses_{method}')
        plt.close(fig_losses)
        
    # Save R Matrix Figure
    if R is not None:
        fig_heatmap = heatmap_plot(R=R)
        save_fig(fig_heatmap,
                 save_dir=save_dir,
                 save_name=f'R_heatmap_{method}')
        plt.close(fig_heatmap)

    plt.close('all')
