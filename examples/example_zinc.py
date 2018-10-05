#!/usr/bin/env python
import urllib.request

import sacred

ex = Experiment("conditional_diversity_networks")

@ex.config
def config():
    workdir = './ex1'
    data_root = '~/.mnist'
    observer = FileStorageObserver.create(str(Path(workdir).resolve() / 'config'))

@ex.capture
def download():
    urls = ['https://github.com/0h-n0/cdn_molecule_pytorch/blob/data/data/250k_rndm_zinc_drugs_clean.smi',
            'https://github.com/0h-n0/cdn_molecule_pytorch/blob/data/data/FDA_drugs.smi']
    for u in urls:
        local_filename, headers = urllib.request.urlretrieve(u)

@ex.capture
def train():
    pass


@ex.automain
def main(_log, epochs):
    _log.info('hello')
    train()
    
