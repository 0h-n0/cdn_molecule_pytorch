from pathlib import Path

import torch

class SmilesDataLoader(object):
    _SMILES_CHARS = [' ',
                     '#', '%', '(', ')', '+', '-', '.', '/',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     '=', '@',
                     'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                     'R', 'S', 'T', 'V', 'X', 'Z',
                     '[', '\\', ']',
                     'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                     't', 'u']
    
    def __init__(self, filename, max_length=50, smiles_chars=None):
        self.filename = filename
        self.max_length = max_length
        self.raw_data = self._preprocess(self._read(filename))
        self.size = len(self.raw_data)
        if smiles_chars is None:
            self.smiles_chars = self._SMILES_CHARS
        else:
            self.smiles_chars = smiles_chars

    def _preprocess(self, raw_data):
        '''
        remove a data whose lenght is less than self.max_length.
        '''
        _raw_data = []
        for smiles in raw_data:
            if len(smiles) < self.max_length:
                _raw_data.append(smiles)
        return _raw_data

    def _read(self, filename):
        raw_data = None
        f = Path(filename).expanduser().resolve()
        with f.open() as fp:
            raw_data = fp.readlines()
        return raw_data
            
    def __getitem__(self, i):
        return torch.LongTensor(self._smiles_to_vec(self.raw_data[i].strip().split()[0]))

    def __len__(self):
        return self.size

    def _smiles_to_vec(self, smiles: str, smiles_chars=None):
        if smiles_chars is None:
            smiles_chars = self.smiles_chars
        input_length = len(smiles)
        if input_length >= self.max_length:
            raise ValueError("input_length >= self.max_length [%d >= %d] ", input_length, self.max_length)
        
        vec = torch.zeros(self.max_length, dtype=torch.long)
        for i, c in enumerate(smiles):
            vec[i] = self._SMILES_CHARS.index(c)
        '''        
        vec = torch.zeros(input_length, max_length, dtype=torch.long)            
        for i, c in enumerate(smiles):
            vec[i, self._SMILES_CHARS.index(c)] = 1
        '''
        return vec
    

if __name__ == '__main__':
    s = SmilesDataLoader('~/workspace/misc/250k_rndm_zinc_drugs_clean.smi')
    print(len(s))



