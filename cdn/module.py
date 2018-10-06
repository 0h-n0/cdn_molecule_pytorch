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

    
class CDN(torch.nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 kernel_sizes=[3, 4, 5, 6],
                 in_channels=[1, 1, 1, 1],
                 out_channels=[128, 128, 128, 128],
                 max_mol_length=50,
                 smiles_chars=None,
                 variational=True,
                 gaussian_samples=300,
                 generation_mode=False,                 
    ):
        super(CDN, self).__init__()
        self.embed = SmilesEmbbeding(embedding_dim, smiles_chars)

        self.num_conv_layers = len(kernel_sizes)
        self.conv_pads = [ None for i in range(self.num_conv_layers)]        
        self.convs = [ None for i in range(self.num_conv_layers)]
        self.max_mol_length = max_mol_length
        self.variational = variational
        self.gaussian_samples = gaussian_samples
        self.generation_mode = generation_mode
        
        _image_size = [max_mol_length, embedding_dim] # H, W
        
        for i in range(self.num_conv_layers):
            ic = in_channels[i]
            oc = out_channels[i]
            k = kernel_sizes[i]
            kernel_shape = (k, embedding_dim)
            self.convs[i] = torch.nn.Conv2d(ic, oc, kernel_shape)
            # tensorflow filter =  [filter_height, filter_width, in_channels, out_channels]
            # TODO: tf.nn.embedding_lookup()
        self.convs = torch.nn.ModuleList(self.convs)
        in_dim  = 128 * sum(map(lambda x: 50 - x + 1, kernel_sizes))
        self.fc = torch.nn.Linear(in_dim, 450)

        self.fc_mean = torch.nn.Linear(450, gaussian_samples)
        self.fc_stddev = torch.nn.Linear(450, gaussian_samples)

        self.decode_layer = torch.nn.Linear(gaussian_samples, 150)
        self.decode_embed = SmilesEmbbeding(embedding_dim, smiles_chars)        
        
    def encode(self, x):
        x = self.embed(x)
        # x shape is B, W, H
        B, W, H = x.shape
        embed_x = x.reshape(B, 1, W, H)
        # x shape is B, C=1, W, H
        conv_flattens = []
        for i in range(self.num_conv_layers):
            x = self.convs[i](embed_x)
            x = F.relu(x)
            conv_flattens.append(x.view(B, -1))
        flatten_x = torch.cat(conv_flattens, dim=1)
        x = F.relu(self.fc(flatten_x))

        if self.variational:
            self.z_mean = self.fc_mean(x)
            self.z_stddev =self.fc_stddev(x)
            latent_loss = 0.5 * torch.sum(self.z_mean**2 + self.z_stddev**2 -
                                          torch.log(self.z_stddev**2) -1,  1)
            self.mean_latent_loss = torch.mean(latent_loss)
            if self.generation_mode:
                h_pool_flat = self.gaussian_samples
            else:
                h_pool_flat = self.z_mean + (self.z_stddev * self.gaussian_samples)
            #h_pool_flat = h_pool_flat.clone()
            
        return h_pool_flat, self.mean_latent_loss

    def decode(self, z, x):
        z = F.relu(self.decode_layer(z))
        embed_x = self.decode_embed(x)
        embed_x_list = torch.split(embed_x, self.max_mol_length, dim=1)
        print(embed_x_list)
        return embed_x_list
    
    def forward(self, x):
        z, mean_latent_loss = self.encode(x)
        z = self.decode(z, x)
        return z, x
    
    
class DiversityLayer(torch.nn.Module):
    def __init__(self):
        super(DiversityLayer, self).__init__()        
        pass
    

if __name__ == '__main__':
    x = torch.LongTensor([[i for i in range(50)], [i for i in range(50)]])
    # TODO: padding

    cdn = CDN()
    y, loss = cdn(x)

    print(cdn)
    print(y.shape)    
