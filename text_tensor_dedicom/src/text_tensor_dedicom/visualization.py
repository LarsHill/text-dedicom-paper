import io
import os
from typing import Dict, Optional, List, Tuple

from tempfile import TemporaryDirectory

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
from torchvision.transforms import ToTensor

sns.set_style("whitegrid")
sns.set_context("paper", rc={"grid.linewidth": 0.5})


tex_fonts = {
    # Use LaTeX to write all text
    "text.latex.preamble": [r'\usepackage{bm}'],
    "font.family": "serif",
    "font.serif": 'cm',
    "text.usetex": True,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)


def df_to_tex(df: pd.DataFrame,
              save_dir: str,
              save_name: str,
              caption: str,
              label: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, save_name), 'w') as f:
        f.write(df.to_latex(index=False,
                            caption=caption,
                            label=label))


def df_to_txt(df: pd.DataFrame,
              save_dir: str,
              save_name: str):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, save_name), 'w') as f:
        f.write(df.to_string(index=False))


def fig_to_tensorboard(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


def save_fig(fig,
             save_dir: str,
             save_name: str):

    paper_dir = os.path.join(save_dir, 'paper_output')
    os.makedirs(paper_dir, exist_ok=True)

    out_dir = os.path.join(save_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    # save as png
    fig.savefig(os.path.join(out_dir, f'{save_name}.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0)

    # Save as pdf
    fig.savefig(os.path.join(paper_dir, f'{save_name}.pdf'), format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def merge_plots(figs: List[plt.figure], figsize=(10, 5)) -> plt.figure:
    with TemporaryDirectory() as tempdir:
        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(tempdir, f'{i}.png'), bbox_inches='tight',
                        pad_inches=0, dpi=300)

        f, axs = plt.subplots(1, len(figs), figsize=figsize, squeeze=False)
        for i in range(len(figs)):
            axs[0][i].imshow(mpimg.imread(os.path.join(tempdir, f'{i}.png')))
        [ax.set_axis_off() for ax in axs.ravel()]

        f.tight_layout(pad=0.1)
        return f


def heatmap_plot(R: np.ndarray,
                 annotate: bool = False,
                 bin_names: List[str] = None) -> plt.figure:
    assert len(R.shape) == 3

    if annotate:
        R = np.round(R, 1)

    if R.shape[0] > 8:
        num_rows = 3
    elif R.shape[0] > 3:
        num_rows = 2
    else:
        num_rows = 1

    num_cols = (R.shape[0] - 1) // num_rows + 1

    figsize = (num_cols * 2, num_rows * 2)

    if bin_names is None:
        bin_names = [None] * R.shape[0]

    max_value = R.max()
    if R.min() < 0:
        min_value = R.min()
    else:
        min_value = 0.

    width_ratios = [1] * num_cols + [0.08]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(num_rows, num_cols + 1, width_ratios=width_ratios)

    for i in range(R.shape[0]):
        row = i // num_cols
        col = i % num_cols

        ax = fig.add_subplot(gs[row, col])
        if i % num_cols == 0 and i // num_cols == (num_rows - 1):
            g = sns.heatmap(data=R[i],
                            vmin=min_value,
                            vmax=max_value,
                            cmap='Blues' if R.min() >= 0 else 'RdBu',
                            square=True,
                            yticklabels=range(1, R.shape[1] + 1),
                            xticklabels=range(1, R.shape[1] + 1),
                            annot=False,
                            annot_kws={"size": 10},
                            cbar=False,
                            cbar_ax=None,
                            ax=ax)
            g.set_ylabel('')
            g.set_xlabel('')
            g.set_title(bin_names[i])
        elif i % num_cols == 0:
            g = sns.heatmap(data=R[i],
                            vmin=min_value,
                            vmax=max_value,
                            cmap='Blues' if R.min() >= 0 else 'RdBu',
                            square=True,
                            yticklabels=range(1, R.shape[1] + 1),
                            xticklabels=range(1, R.shape[1] + 1),
                            annot=False,
                            annot_kws={"size": 10},
                            cbar=False,
                            cbar_ax=None,
                            ax=ax)
            g.set_ylabel('')
            g.set_xlabel('')
            g.set_xticks([])
            g.set_title(bin_names[i])
        elif i == R.shape[0] - 1:
            axcb = fig.add_subplot(gs[:, -1])
            g = sns.heatmap(data=R[i],
                            vmin=min_value,
                            vmax=max_value,
                            cmap='Blues' if R.min() >= 0 else 'RdBu',
                            square=True,
                            yticklabels=range(1, R.shape[1] + 1),
                            xticklabels=range(1, R.shape[1] + 1),
                            annot=False,
                            annot_kws={"size": 10},
                            cbar=True,
                            cbar_ax=axcb,
                            cbar_kws={"shrink": 1.},
                            ax=ax)
            g.set_ylabel('')
            g.set_xlabel('')
            g.set_yticks([])
            g.set_title(bin_names[i])
        elif i // num_cols == (num_rows - 1):
            g = sns.heatmap(data=R[i],
                            vmin=min_value,
                            vmax=max_value,
                            cmap='Blues' if R.min() >= 0 else 'RdBu',
                            square=True,
                            yticklabels=range(1, R.shape[1] + 1),
                            xticklabels=range(1, R.shape[1] + 1),
                            annot=False,
                            annot_kws={"size": 10},
                            cbar=False,
                            ax=ax)
            g.set_ylabel('')
            g.set_xlabel('')
            g.set_yticks([])
            g.set_title(bin_names[i])
        else:
            g = sns.heatmap(data=R[i],
                            vmin=min_value,
                            vmax=max_value,
                            cmap='Blues' if R.min() >= 0 else 'RdBu',
                            square=True,
                            yticklabels=range(1, R.shape[1] + 1),
                            xticklabels=range(1, R.shape[1] + 1),
                            annot=False,
                            annot_kws={"size": 10},
                            cbar=False,
                            cbar_ax=None,
                            ax=ax)
            g.set_ylabel('')
            g.set_xlabel('')
            g.set_yticks([])
            g.set_xticks([])
            g.set_title(bin_names[i])

    fig.tight_layout()
    return fig


def scatterplot_colored(A_fit: np.array,
                        id_to_word: Dict[int, str],
                        topics_all: Optional[Dict[str,
                                                  List[Tuple[str, int, np.array]]]] = None,
                        titles: Optional[List[str]] = None,
                        word_to_titles: Optional[Dict[str, List[str]]] = None):
    all_colors = ['#002C63',  # dark blue
                  '#ff9c00',  # orange/yellow
                  '#d40000',  # red
                  "#d3d10f",  # yellow
                  "#bf00d6",  # purple
                  "#19c631",  # green
                  '#33eff2',  # light blue
                  '#bd7e00',  # brown
                  '#00910c',  # dark green
                  '#1828d9',  # Blue
                  '#ff0f63',  # magenta red,
                  '#ff00d9'   # pink
                  ]
    if titles and not topics_all:
        title2color = {f'{title}': all_colors[i] for i, title in enumerate(list(titles))}
        colors = []
        indices = []
        for i in range(A_fit.shape[0]):
            word = ''.join(id_to_word[i])
            if len(word_to_titles[word]) == 1:
                title = word_to_titles[word][0]
                colors.append(title2color[f'{title}'])
                indices.append(i)

        fig = plt.figure(figsize=(2, 2))
        plt.scatter(x=A_fit[indices, 0], y=A_fit[indices, 1], edgecolors='none', c=colors, s=1, alpha=0.2)

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
        plt.scatter(x=A_fit[indices, 0], y=A_fit[indices, 1], edgecolors='none', c=colors, s=1, alpha=0.2)

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
    colors = ['#ff9c00', '#002C63', '#d40000']
    for i, (key, value) in enumerate(losses.items()):
        label = key.replace('_loss', '').upper()
        epochs = [int(epoch) for epoch in value.keys()]
        plt.plot(epochs,
                 list(value.values()),
                 color=colors[i],
                 label=label,
                 linestyle='-' if 'dedicom' in key else '--',
                 linewidth=1.5)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    return fig
