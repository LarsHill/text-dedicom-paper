from collections import MutableMapping
from copy import deepcopy
from itertools import product
import os
import random
from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch


def set_device(device: str):
    global _DEVICE
    _DEVICE = torch.device(device)


def get_device() -> torch.device:
    return _DEVICE


def set_seed_number(seed: int):
    global _SEED
    _SEED = seed


def set_seeds():
    torch.manual_seed(_SEED)
    random.seed(_SEED)
    np.random.seed(_SEED)


def get_balanced_devices(count: int,
                         no_cuda: bool = False) -> List[str]:
    if not no_cuda and torch.cuda.is_available():
        devices = [f'cuda:{id_}' for id_ in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    factor = int(count / len(devices))
    remainder = count % len(devices)
    devices = devices * factor + devices[:remainder]
    return devices


def batch_sequence(seq: Sequence, batch_size: int) -> Tuple:
    return tuple(seq[pos: pos + batch_size] for pos in range(0, len(seq), batch_size))


def create_new_run_dir(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'output'), exist_ok=True)
    return run_dir


def find_list_in_dict(obj: Dict, param_grid: List) -> List:
    for key in obj:
        if isinstance(obj[key], list):
            param_grid.append([val for val in obj[key]])
        elif isinstance(obj[key], dict):
            find_list_in_dict(obj[key], param_grid)
        else:
            continue
    return param_grid


def replace_list_in_dict(obj: Dict, obj_copy: Dict, comb: Tuple, counter: List) -> Tuple[Dict, List]:
    for key, key_copy in zip(obj, obj_copy):
        if isinstance(obj[key], list):
            obj_copy[key_copy] = comb[len(counter)]
            counter.append(1)
        elif isinstance(obj[key], dict):
            replace_list_in_dict(obj[key], obj_copy[key_copy], comb, counter)
        else:
            continue
    return obj_copy, counter


def dict_to_list_of_strings(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(dict_to_list_of_strings(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return [str(k + sep + str(v)) for k, v in items]


def split_gs_config(config_grid_search: Dict) -> List[Dict]:
    param_grid = []
    param_grid = find_list_in_dict(config_grid_search, param_grid)
    config_copy = deepcopy(config_grid_search)
    individual_configs = []
    for comb in product(*param_grid):
        counter = []
        individual_config = replace_list_in_dict(config_grid_search, config_copy, comb, counter)[0]
        individual_config = deepcopy(individual_config)
        individual_configs.append(individual_config)
    return individual_configs


def filter_run_configs(run_configs: List[Dict],
                       blacklist: List[List[str]]) -> List[Dict]:
    for comb in blacklist:
        for item in comb:
            flag = False
            for cfg in run_configs:
                if item in dict_to_list_of_strings(cfg):
                    flag = True
            if not flag:
                raise ValueError(f'There is a typo in the blacklist. Check "{item}".')

    filtered_run_configs = []
    for cfg in run_configs:
        cfg_flattened = dict_to_list_of_strings(cfg)
        flag = False
        for comb in blacklist:
            if all(item in cfg_flattened for item in comb):
                flag = True
        if not flag:
            filtered_run_configs.append(cfg)
    return filtered_run_configs


def create_run_configs(config_grid_search: Dict) -> List[Dict]:
    try:
        blacklist = config_grid_search.pop('blacklist')
    except KeyError:
        blacklist = None

    run_configs = split_gs_config(config_grid_search)

    if blacklist:
        run_configs = filter_run_configs(run_configs=run_configs,
                                         blacklist=blacklist)
    return run_configs
