import io
import os
from typing import Dict, Optional, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
from torchvision.transforms import ToTensor

plt.rcParams.update({
    "text.usetex": True
})

sns.set_style("whitegrid")
sns.set_context("paper", rc={"grid.linewidth": 0.5})


def df_to_txt(df: pd.DataFrame,
              save_dir: str,
              save_name: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, save_name), 'w') as f:
        f.write(df.to_string(index=False))


def fig_to_tensorboard(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


def save_fig(fig,
             save_dir: str,
             save_name: str):
    out_dir = os.path.join(save_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f'{save_name}.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def heatmap_plot(R: np.array, annotate=False):
    if annotate:
        R = np.round(R, 1)

    fig = plt.figure(figsize=(3, 2))
    ax = sns.heatmap(data=R,
                     vmin=-np.abs(R).max(),
                     vmax=np.abs(R).max(),
                     center=0,
                     cmap='RdBu',
                     square=True,
                     yticklabels=range(1, R.shape[0] + 1),
                     xticklabels=range(1, R.shape[0] + 1),
                     annot=annotate,
                     annot_kws={"size": 6})
    ax.set_title('R')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.add_axes(ax)
    fig.tight_layout()
    return fig


def scatterplot_colored(A_fit: np.array,
                        id2word: Dict[int, str],
                        topics_all: Optional[Dict[str,
                                                  List[Tuple[str, int, np.array]]]] = None,
                        titles: Optional[List[str]] = None,
                        word2titles: Optional[Dict[str, List[str]]] = None):
    all_colors = ['#002C63', '#ff9c00', '#d40000', "#d3d10f", "#bf00d6", "#19c631", "#16b9ef", '#bd7e00', '#00910c']
    if titles and not topics_all:
        title2color = {f'{title}': all_colors[i] for i, title in enumerate(list(titles))}
        colors = []
        indices = []
        for i in range(A_fit.shape[0]):
            word = ''.join(id2word[i])
            if len(word2titles[word]) == 1:
                title = word2titles[word][0]
                colors.append(title2color[f'{title}'])
                indices.append(i)

        fig = plt.figure(figsize=(2, 2))
        plt.scatter(x=A_fit[indices, 0], y=A_fit[indices, 1], edgecolors='none', c=colors, s=10, alpha=0.2)

        legend_patches = []
        for title, color in title2color.items():
            legend_patches.append(mpatches.Patch(color=color, label=title))
        plt.legend(handles=legend_patches,
                   fontsize='xx-small',
                   loc='lower left',
                   bbox_to_anchor=(-0.03, 1.02, 1., 0.2),
                   borderpad=0.6,
                   columnspacing=1,
                   ncol=1)
    elif topics_all and not titles:
        title2color = {f'{title}': all_colors[i] for i, title in enumerate(list(topics_all.keys()))}
        colors = []
        indices = []
        for title, words in topics_all.items():
            for word in words:
                indices.append(word[1])
                colors.append(title2color[f'{title}'])

        fig = plt.figure(figsize=(2, 2))
        plt.scatter(x=A_fit[indices, 0], y=A_fit[indices, 1], edgecolors='none', c=colors, s=10, alpha=0.2)

        legend_patches = []
        for title, color in title2color.items():
            legend_patches.append(mpatches.Patch(color=color, label=title))
        plt.legend(handles=legend_patches,
                   fontsize='xx-small',
                   loc='lower left',
                   bbox_to_anchor=(-0.16, 1.02, 1., 0.2),
                   borderpad=0.6,
                   columnspacing=1,
                   ncol=3)
    else:
        raise ValueError('Titles or topics all need to be provided.')
    return fig


def losses_plot(losses: Dict[str, Dict[int, float]]):
    fig = plt.figure(figsize=(5, 2.5))
    plt.plot(list(losses['dedicom_loss'].keys()),
             list(losses['dedicom_loss'].values()),
             color='#002C63',
             label='DEDICOM ',
             linewidth=1.5)
    plt.plot(list(losses['dedicom_loss'].keys()),
             [losses['nmf_loss']] * len(losses['dedicom_loss'].keys()),
             linestyle='--',
             color='#ff9c00',
             label='NMF ',
             linewidth=1.5)
    plt.plot(list(losses['dedicom_loss'].keys()),
             [losses['svd_loss']] * len(losses['dedicom_loss'].keys()),
             linestyle='--',
             color='#d40000',
             label='SVD ',
             linewidth=1.5)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    return fig
