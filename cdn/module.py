import torch
import torch.nn.functional as F

def one_hot(x, size):
    B, T = x.shape    
    inp = torch.LongTensor(B, T) % size    
    inp_ = torch.unsqueeze(inp, 2)    
    one_hot = torch.FloatTensor(B, T, size).zero_()
    one_hot.scatter_(2, inp_, 1)
    return one_hot

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
            self.vocab_size = len(self.smiles_chars)            
        else:
            self.smiles_chars =  self._SMILES_CHARS
            self.num_embeddings = len(self._SMILES_CHARS)
            self.vocab_size = len(self._SMILES_CHARS)                        

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
                 test_mode=False
    ):
        super(CDN, self).__init__()
        self.embed = SmilesEmbbeding(embedding_dim, smiles_chars)
        self.vocab_size = self.embed.vocab_size
        self.num_conv_layers = len(kernel_sizes)
        self.conv_pads = [ None for i in range(self.num_conv_layers)]        
        self.convs = [ None for i in range(self.num_conv_layers)]
        self.max_mol_length = max_mol_length
        self.variational = variational
        self.gaussian_samples = gaussian_samples
        self.generation_mode = generation_mode
        self.test_mode = test_mode
        
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

        self.lstm_cell = torch.nn.LSTMCell(embedding_dim, 150)
        self.fc_output = torch.nn.Linear(150, self.vocab_size)
        
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
                
            h_pool_flat = h_pool_flat.clone()
            
        return h_pool_flat, self.mean_latent_loss

    def decode(self, z, x):
        z = F.relu(self.decode_layer(z))
        embed_x = self.decode_embed(x)
        embed_x_list = torch.chunk(embed_x, self.max_mol_length, dim=1)
        assert len(embed_x_list) == self.max_mol_length, 'Something wrong'
        
        decoder_inputs_list = [torch.squeeze(i, dim=1) for i in embed_x_list]
        #decoder_inputs_list = embed_x_list
        
        temp_logits = []
        lstm_outputs = []
        all_symbols = []
        symbol = torch.ones(1)        
        for i in range(self.max_mol_length):
            if not self.test_mode or i == 0:
                if i == 0:
                    output, state = self.lstm_cell(decoder_inputs_list[i], (z, z))
                else:
                    output, state = self.lstm_cell(decoder_inputs_list[i], (state, state))
            else:
                next_decoder_input, symbol = pick_next_argmax(temp_logits[-1], i)
                next_decoder_input = tf.squeeze(next_decoder_input, axis=1)
                output, state = self.lstm_cell(next_decoder_input, state)
                
            temp_logits.append(self.fc_output(output))
            lstm_outputs.append(output)
            if i > 0:
                all_symbols.append(symbol)
            if i == self.max_mol_length - 1 and self.test_mode:
                all_symbols.append(pick_next_argmax(temp_logits[-1], i)[1])
        decoder_logits = torch.t(torch.stack(temp_logits))

        return decoder_logits
    
    def forward(self, x):
        z, mean_latent_loss = self.encode(x)
        decoder_logits = self.decode(z, x)
        return mean_latent_loss, decoder_logits
    
    def loss_acc(self, x, logits, latent_loss):
        onehot_target = one_hot(x, self.vocab_size)
        losses = F.binary_cross_entropy_with_logits(logits, onehot_target)
        ce_loss = torch.mean(losses)
        total_loss = ce_loss + .00001 * latent_loss
        
        prediction = torch.argmax(decoder_logits, 2)
        x_target = x
        correct_predictions = torch.eq(prediction, x_target)
        acc = torch.mean(correct_predictions.type(torch.FloatTensor))
        return total_loss, acc
        
if __name__ == '__main__':
    x = torch.LongTensor([[i for i in range(50)], [i for i in range(50)]])
    # TODO: padding
    cdn = CDN()
    latent_loss, decoder_logits = cdn(x)
    loss, acc = cdn.loss_acc(x, decoder_logits, latent_loss)
