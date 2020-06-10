from argparse import ArgumentParser
from datetime import datetime
from itertools import repeat
import json
from multiprocessing import Pool, set_start_method
import os
import pickle
from typing import Dict

import torch
from tqdm import tqdm
import yaml

from text_dedicom.data_factory import get_input_matrix_from_wiki_data
from text_dedicom.evaluation import generate_paper_output
from text_dedicom.models.dedicom import run_dedicom
from text_dedicom.models.lda import run_lda
from text_dedicom.models.nmf import run_nmf
from text_dedicom.models.svd import run_truncated_svd
from text_dedicom.utils import set_device, set_seed_number, get_device, create_new_run_dir, create_run_configs, get_balanced_devices, batch_sequence


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        default='../../config.yaml',
                        help='Path to config file (yaml format).',
                        type=str)
    parser.add_argument('--num-processes',
                        default=1,
                        type=int)
    parser.add_argument('-v',
                        '--verbose',
                        action='store_false')
    return parser.parse_args()


def run_pipeline(config: Dict,
                 device: str,
                 verbose: bool = False):
    out_dir = config.get('out_dir')
    wiki_articles = config.get('wiki_articles')
    matrix_type = config.get('matrix_type')
    preprocessing = config.get('preprocessing')
    window_size = config.get('window_size')
    a_constraint = config.get('a_constraint')
    seed = config.get('seed')
    k = config.get('k')
    lr_a = config.get('lr_a')
    lr_r = config.get('lr_r')
    num_epochs = config.get('num_epochs')
    evaluate_every = config.get('evaluate_every')

    set_seed_number(seed)
    set_device(device)

    run_dir = f'{str(wiki_articles)} @ {matrix_type} @ {a_constraint} @ k={k} @ lr_a={lr_a} @ lr_r={lr_r}'

    out_dir = create_new_run_dir(os.path.join(out_dir, run_dir))
    json.dump(config, open(os.path.join(out_dir, 'config.json'), 'w'))

    print('## Prepare Data... ##')
    matrix, id2word, _, word2titles = get_input_matrix_from_wiki_data(wiki_articles=wiki_articles,
                                                                      preprocessing=preprocessing,
                                                                      window_size=window_size,
                                                                      matrix_type=matrix_type)
    pickle.dump(matrix, open(os.path.join(out_dir, 'input_matrix.p'), 'wb'))

    print()
    print()
    print('## Train Models... ##')
    print('LDA')
    W_lda, H_lda = run_lda(M=matrix, k=k, save_dir=out_dir)

    print('NMF')
    W_nmf, H_nmf = run_nmf(M=matrix, k=k, save_dir=out_dir)

    print('SVD')
    U_sigma, V = run_truncated_svd(M=matrix, k=k, save_dir=out_dir)

    print('DEDICOM')
    A, R, losses = run_dedicom(save_dir=out_dir,
                               M=torch.tensor(matrix).to(get_device()).type(torch.float),
                               k=k,
                               n=matrix.shape[0],
                               lr_a=lr_a,
                               lr_r=lr_r,
                               num_epochs=num_epochs,
                               verbose=verbose,
                               evaluate_every=evaluate_every,
                               id2word=id2word,
                               a_constraint=a_constraint)

    # LDA
    print('## Generate Output... ##')
    print('LDA')
    generate_paper_output(M=H_lda.T,
                          id2word=id2word,
                          word2titles=word2titles,
                          titles=wiki_articles,
                          save_dir=out_dir,
                          method='lda_h')
    generate_paper_output(M=W_lda,
                          id2word=id2word,
                          word2titles=word2titles,
                          titles=wiki_articles,
                          save_dir=out_dir,
                          method='lda_w')

    # NMF
    print('NMF')
    generate_paper_output(M=H_nmf.T,
                          id2word=id2word,
                          word2titles=word2titles,
                          titles=wiki_articles,
                          save_dir=out_dir,
                          method='nmf_h')
    generate_paper_output(M=W_nmf,
                          id2word=id2word,
                          word2titles=word2titles,
                          titles=wiki_articles,
                          save_dir=out_dir,
                          method='nmf_w')

    # SVD
    print('SVD')
    generate_paper_output(M=V.T,
                          id2word=id2word,
                          word2titles=word2titles,
                          titles=wiki_articles,
                          save_dir=out_dir,
                          method='svd_h')
    generate_paper_output(M=U_sigma,
                          id2word=id2word,
                          word2titles=word2titles,
                          titles=wiki_articles,
                          save_dir=out_dir,
                          method='svd_w')

    # DEDICOM
    print('DEDICOM')
    generate_paper_output(M=A,
                          R=R,
                          losses=losses,
                          id2word=id2word,
                          word2titles=word2titles,
                          titles=wiki_articles,
                          save_dir=out_dir,
                          method='dedicom')


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    single_run_configs = create_run_configs(config)

    start = datetime.now()

    set_start_method('spawn', force=True)
    for batch_run_configs in tqdm(batch_sequence(single_run_configs, batch_size=args.num_processes)):
        devices = get_balanced_devices(count=len(batch_run_configs))

        with Pool(processes=len(batch_run_configs)) as pool:
            pool.starmap(run_pipeline, zip(batch_run_configs,
                                           devices,
                                           repeat(args.verbose)))

    end = datetime.now()
    print(end - start)


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    main()
