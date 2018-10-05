import numpy as np


_SMILES_CHARS = [' ',
                 '#', '%', '(', ')', '+', '-', '.', '/',
                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 '=', '@',
                 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                 'R', 'S', 'T', 'V', 'X', 'Z',
                 '[', '\\', ']',
                 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                 't', 'u']  ## 56


def smiles_to_vec(smiles: str, smiles_chars=_SMILES_CHARS):
    max_length = len(smiles_chars)
    input_length = len(smiles)
    vec = np.zeros((max_length, input_length), dtype=np.int32)
    for i, c in enumerate(smiles):
        vec[smiles_chars.index(c), i] = 1
    return vec

if __name__ == '__main__':
    print(smiles_to_vec("ccccc6c"))

