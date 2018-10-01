#!/usr/bin/env python


import urllib.request

def download():
    urls = ['https://github.com/0h-n0/cdn_molecule_pytorch/blob/data/data/250k_rndm_zinc_drugs_clean.smi',
             'https://github.com/0h-n0/cdn_molecule_pytorch/blob/data/data/FDA_drugs.smi']

    for u in urls:
        local_filename, headers = urllib.request.urlretrieve(u)



if __name__ == '__main__':
    download()
    
