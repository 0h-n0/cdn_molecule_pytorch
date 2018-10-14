#!/usr/bin/env python
import shutil
import urllib.request
from pathlib import Path

import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver

import cdn
import cdn.data


ex = Experiment("conditional_diversity_networks")

@ex.config
def config():
    workdir = './ex1'
    data_root = './'
    observer = FileStorageObserver.create(str(Path(workdir).resolve() / 'config'))
    filenames = ['250k_rndm_zinc_drugs_clean.smi',
                 'FDA_drugs.smi']
    del observer
    
@ex.capture
def download(_log, data_root, filenames):
    urls = ['https://github.com/0h-n0/cdn_molecule_pytorch/raw/data/data/250k_rndm_zinc_drugs_clean.smi',
            'https://github.com/0h-n0/cdn_molecule_pytorch/raw/data/data/FDA_drugs.smi']
    
    for u, fname in zip(urls, filenames):
        if (Path(data_root) / fname).exists():
            _log.info("SKIP: {} already exists.".format(str(Path(data_root) / fname)))
            continue
        local_filename, headers = urllib.request.urlretrieve(u)
        src = Path(local_filename)
        shutil.copy(str(src), str(Path(data_root) / fname))

@ex.capture
def get_iterators(_log, data_root, filenames):
    train_iterator = cdn.data.SmilesDataLoader(str(Path(data_root) / filenames[0]))
    test_iterator = cdn.data.SmilesDataLoader(str(Path(data_root) / filenames[1]))    
    return train_iterator, test_iterator
        
@ex.capture
def train(_log):
    _log.info('train')    
    train_iterator, test_iterator = get_iterators()

@ex.automain
def main(_log):
    _log.info('main')
    download()
    train()
    
