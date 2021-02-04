from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from itertools import repeat
import json
from multiprocessing import Pool, set_start_method
import os
from typing import Dict, List
import pickle

import torch
from tqdm import tqdm
import yaml

from text_tensor_dedicom import project_path
from text_tensor_dedicom.binning import bin_articles
from text_tensor_dedicom.filter_corpus import filter_sections, filter_dates
from text_tensor_dedicom.data_classes import Corpus
from text_tensor_dedicom.evaluation import generate_paper_output
from text_tensor_dedicom.featurizing import Featurizer
from text_tensor_dedicom.models.tnmf import run_tnmf
from text_tensor_dedicom.models.dedicom import run_dedicom
from text_tensor_dedicom.preprocessing import Preprocessor
from text_tensor_dedicom.utils import (set_device, set_seed_number,
                                       create_new_run_dir, create_run_configs,
                                       get_balanced_devices, batch_sequence, hash_from_dict,
                                       removekeys)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        default='configs/config_nyt_news.yaml',
                        choices=['configs/config_nyt_news.yaml',
                                 'configs/config_wiki.yaml',
                                 'configs/config_amazon_reviews.yaml'],
                        help='Path to config file (yaml format).',
                        type=str)
    parser.add_argument('--num-processes',
                        default=1,
                        type=int)
    parser.add_argument('-v',
                        '--verbose',
                        action='store_false')
    parser.add_argument('--no-cache',
                        action='store_true')
    parser.add_argument('--no-cuda',
                        action='store_true')
    return parser.parse_args()


def run_pipeline(config: Dict,
                 device: str,
                 verbose: bool = False,
                 no_cache: bool = False):
    in_file = config.get('in_file')
    out_dir = config.get('out_dir')
    cache_dir = config.get('cache_dir')
    bin_by = config.get('bin_by')
    bin_size = config.get('bin_size')
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    filtered_sections = config.get('filtered_sections')
    preprocessing = config.get('preprocessing')
    matrix_type = config.get('matrix_type')
    window_size = config.get('window_size')
    vocab_size = config.get('vocab_size')
    dim_topic = config.get('dim_topic')
    lr_a = config.get('lr_a')
    lr_a_scaling = config.get('lr_a_scaling', 1)
    lr_r = config.get('lr_r')
    a_updates_per_epoch = config.get('a_updates_per_epoch', 1)
    r_updates_per_epoch = config.get('r_updates_per_epoch', 1)
    num_epochs = config.get('num_epochs')
    evaluate_every = config.get('evaluate_every')
    seed = config.get('seed')
    run_name = config.get('run_name', None)
    one_topic_per_word = config.get('one_topic_per_word', False)
    method = config.get('method', 'mult')

    set_seed_number(seed)
    set_device(device)

    in_file = os.path.join(project_path, in_file)

    run_dir = run_name if run_name is not None else 'run'

    out_dir = create_new_run_dir(os.path.join(out_dir, run_dir))
    json.dump(config, open(os.path.join(out_dir, 'config.json'), 'w'))
    os.makedirs(cache_dir, exist_ok=True)

    keys_no_cache = ['out_dir', 'cache_dir', 'run_name', 'nmf_init',
                     'dim_topic', 'lr_a', 'lr_a_scaling', 'lr_r', 'a_updates_per_epoch', 'r_updates_per_epoch',
                     'a_constraint', 'r_constraint',
                     'num_epochs', 'evaluate_every', 'seed',
                     'one_topic_per_word', 'method']
    cache_file = hash_from_dict(removekeys(config, keys_no_cache))
    cache_file = os.path.join(cache_dir, cache_file)

    if os.path.exists(cache_file) and not no_cache:
        print(f'========= Read cached data...           ===========\n')
        print(f'Cache file:\n{cache_file}\n')
        input_tensor, id_to_word, vocab_len, bin_names, word_to_bins = pickle.load(open(cache_file, 'rb'))

    else:
        print(f'========= Parse data...                 ===========\n')
        corpus = Corpus.from_json(path=in_file)

        if filtered_sections is not None:
            corpus = filter_sections(corpus=corpus, filtered_sections=filtered_sections)
        if start_date and end_date:
            corpus = filter_dates(corpus=corpus, start_date=start_date, end_date=end_date)
        if bin_by is not None:
            corpus.bins = bin_articles(corpus.articles, bin_by=bin_by, bin_size=bin_size)

        print(f'Parsed into {len(corpus.bins)} bins:')
        for bin_ in corpus.bins:
            print(bin_.id_, bin_)
        bin_names: List[str] = [str(bin_.id_) for bin_ in corpus.bins]

        print(f'========= Preprocess data...            ===========\n')
        preprocessor = Preprocessor(pipeline=preprocessing)
        corpus_processed = preprocessor.process(corpus=corpus)

        unique_total_words = set()
        total_words = []
        average_unique_total_words_article = []
        average_total_words_article = []
        word_counter = Counter()
        for bin_ in corpus_processed:
            for article in bin_.articles:
                unique_total_words.update(set(article.value_processed.split()))
                total_words.extend(article.value_processed.split())
                average_total_words_article.append(len(article.value_processed.split()))
                average_unique_total_words_article.append(len(set(article.value_processed.split())))
                word_counter.update(article.value_processed.split())

        print(f'total words: {len(total_words)}\n'
              f'unique total words: {len(unique_total_words)}\n'
              f'average total words article: '
              f'{round(sum(average_total_words_article) / len(average_total_words_article), 2)}\n'
              f'average unique words article: '
              f'{round(sum(average_unique_total_words_article) / len(average_unique_total_words_article), 2)}\n'
              f'Cutoff threshold most frequent words: '
              f'{word_counter.most_common(10000)[0]} {word_counter.most_common(10000)[-1]}')

        print(f'========= Featurize data...             ===========\n')
        featurizer = Featurizer(corpus=corpus_processed,
                                window_size=window_size,
                                matrix_type=matrix_type,
                                vocab_size=vocab_size)
        featurizer.assign_train_valid_test_set(train_split=1., valid_split=0.)
        input_tensor, word_to_bins = featurizer.featurize()
        id_to_word = featurizer.id_to_word
        vocab_len = featurizer.vocab_len

        print(f'========= Cache data...                 ===========\n')
        if not os.path.exists(cache_file) and not no_cache:
            print(f'Cache file:\n{cache_file}\n')
            pickle.dump((input_tensor, id_to_word, vocab_len, bin_names, word_to_bins),
                        open(cache_file, 'wb'), protocol=4)

    print(f'========= Train TNMF model... ===========\n')
    W, H = run_tnmf(M=torch.tensor(input_tensor).type(torch.float),
                    save_dir=out_dir,
                    dim_topic=dim_topic,
                    dim_vocab=vocab_len,
                    dim_time=input_tensor.shape[0],
                    num_epochs=num_epochs)

    generate_paper_output(M=H[0].T,
                          R=None,
                          id_to_word=id_to_word,
                          word_to_titles=word_to_bins,
                          save_dir=out_dir,
                          method='TNMF',
                          bin_names=bin_names)

    print(f'========= Train tensor dedicom model... ===========\n')
    A, R, losses, writer = run_dedicom(save_dir=out_dir,
                                       M=torch.tensor(input_tensor).type(torch.float),
                                       dim_vocab=vocab_len,
                                       dim_topic=dim_topic,
                                       dim_time=input_tensor.shape[0],
                                       lr_a=lr_a,
                                       lr_a_scaling=lr_a_scaling,
                                       lr_r=lr_r,
                                       a_updates_per_epoch=a_updates_per_epoch,
                                       r_updates_per_epoch=r_updates_per_epoch,
                                       num_epochs=num_epochs,
                                       verbose=verbose,
                                       evaluate_every=evaluate_every,
                                       id_to_word=id_to_word,
                                       bin_names=bin_names,
                                       one_topic_per_word=one_topic_per_word,
                                       method=method)

    print(f'========= Generate paper output... ===========\n')

    pickle.dump((A, R), open(os.path.join(out_dir, 'tensors.p'), 'wb'))

    generate_paper_output(M=A,
                          R=R,
                          losses=losses,
                          id_to_word=id_to_word,
                          word_to_titles=word_to_bins,
                          writer=writer,
                          save_dir=out_dir,
                          method='dedicom',
                          bin_names=bin_names)
    writer.close()


def main():
    args = parse_args()
    config_path = os.path.join(project_path, args.config)
    config = yaml.safe_load(open(config_path, 'r'))
    single_run_configs = create_run_configs(config)

    if args.num_processes > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    start = datetime.now()

    if len(single_run_configs) > 1:
        set_start_method('spawn', force=True)
        batches = batch_sequence(single_run_configs, batch_size=args.num_processes)
        if len(batches) == 1:
            pass
        for batch_run_configs in tqdm(batches):
            devices = get_balanced_devices(count=len(batch_run_configs), no_cuda=args.no_cuda)

            with Pool(processes=len(batch_run_configs)) as pool:
                pool.starmap(run_pipeline, zip(batch_run_configs,
                                               devices,
                                               repeat(args.verbose),
                                               repeat(args.no_cache)))
    else:
        config = single_run_configs[0]
        device = get_balanced_devices(count=1, no_cuda=args.no_cuda)[0]
        run_pipeline(config, device, args.verbose, args.no_cache)

    end = datetime.now()
    print(end - start)


if __name__ == '__main__':
    main()
