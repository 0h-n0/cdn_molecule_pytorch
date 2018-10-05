import torch
import torch.nn.functional as F

class SmilesEmbbeding(torch.nn.Module):
    _SMILES_CHARS = [' ',
                     '#', '%', '(', ')', '+', '-', '.', '/',
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     '=', '@',
                     'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                     'R', 'S', 'T', 'V', 'X', 'Z',
                     '[', '\\', ']',
                     'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                     't', 'u']  ## 56
    
    def __init__(self,
                 embedding_dim,
                 smiles_chars=None,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None):
        super(SmilesEmbbeding, self).__init__()
        self.embedding_dim = embedding_dim
        
        if smiles_chars is not None:
            self.smiles_chars = smiles_chars            
            self.num_embeddings = len(self.smiles_chars)
        else:
            self.smiles_chars =  self._SMILES_CHARS
            self.num_embeddings = len(self._SMILES_CHARS)

        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._weight = _weight

        self.e = torch.nn.Embedding(self.num_embeddings,
                                    embedding_dim,
                                    padding_idx,
                                    max_norm,
                                    norm_type,
                                    scale_grad_by_freq,
                                    sparse,
                                    _weight)
                                                
    def forward(self, x):
        return self.e(x)

    
class Encorder(torch.nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 kernel_sizes=[3, 4, 5, 6],
                 in_channels=[1, 1, 1, 1],
                 out_channels=[128, 128, 128, 128],
                 max_length=50,
                 smiles_chars=None,
    ):
        super(Encorder, self).__init__()
        self.embed = SmilesEmbbeding(embedding_dim, smiles_chars)

        self.num_conv_layers = len(kernel_sizes)
        self.conv_pads = [ None for i in range(self.num_conv_layers)]        
        self.convs = [ None for i in range(self.num_conv_layers)]
        _image_size = [max_length, embedding_dim] # H, W
        
        for i in range(self.num_conv_layers):
            ic = in_channels[i]
            oc = out_channels[i]
            k = kernel_sizes[i]
            pad_along_height = k - 1 # filter_height - strides[1]
            pad_along_width = k - 1 # filter_width - strides[2]
            _left = pad_along_width // 2
            _top = pad_along_height // 2
            _right = pad_along_width - _left
            _bottom = pad_along_height - _top
            _padding = (_left, _right, _top, _bottom)
            self.conv_pads[i] = \
                            torch.nn.ConstantPad2d(padding=_padding, value=0)
            # this padding is compatible with Tensorflow's padding='valid'
            kernel_shape = (k, embedding_dim)
            self.convs[i] = torch.nn.Conv2d(ic, oc, kernel_shape)
            # tensorflow filter =  [filter_height, filter_width, in_channels, out_channels]
            # TODO: tf.nn.embedding_lookup()
            
            
    def forward(self, x):
        x = self.embed(x)
        # x shape is B, W, H
        B, W, H = x.shape
        embed_x = x.reshape(B, 1, W, H)
        # x shape is B, C=1, W, H
        print(embed_x.shape)
        conv_flattens = []
        for i in range(self.num_conv_layers):
            padded_x = self.conv_pads[i](self.convs[i](embed_x))
            x = F.relu(padded_x)
            print(x.shape)
            conv_flattens.append(x.view(B, -1))
            print(x.view(B, -1).shape)
        return x

    
class Decorder(torch.nn.Module):
    def __init__(self):
        super(Decorder, self).__init__()                
        pass

    
class DiversityLayer(torch.nn.Module):
    def __init__(self):
        super(DiversityLayer, self).__init__()        
        pass
    
class CDN(torch.nn.Module):
    def __init__(self):
        super(CDN, self).__init__()



if __name__ == '__main__':
    x = torch.LongTensor([[i for i in range(50)], [i for i in range(50)]])
    # TODO: padding

    e = Encorder()
    y = e(x)
    print(y.shape)
    print(e)
    
