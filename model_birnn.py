# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

Required parameters:
    - device to run
    - N_properties (Dimension of label) any additionnal topic we would like to predict from the latent representation
    - N_categories : number of tweet categories / topics 


"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from utils import Variable, BeamSearchNode


from queue import PriorityQueue


class tweetVAE(nn.Module):
    def __init__(self, **kwargs ):
        super(tweetVAE, self).__init__()
        
        self.kappa=1
        self.max_len=kwargs['MAX_LEN']
        self.glove_embeddings = kwargs['weights_matrix']
        
        self.latent_size= 100
        self.h_size=400
        self.device=kwargs['device']
        
        
        # For multitask properties prediction (topic, sentiment, etc...)
        self.N_properties=kwargs['N_properties']
        self.N_topics = kwargs['N_topics']
        
        # Embedding  (GloVe, frozen)
        self.voc_size, self.embedding_dim = self.glove_embeddings.shape
        
        self.emb_layer = nn.Embedding(self.voc_size , self.embedding_dim)
        self.emb_layer.load_state_dict({'weight': self.glove_embeddings})
        self.emb_layer.weight.requires_grad = False
        
        
        
        # ENCODER
        self.birnn = torch.nn.GRU(input_size=self.embedding_dim, hidden_size=self.h_size, num_layers=1,
                                  batch_first=True, bidirectional=True)
        
        # Latent mean and variance
        self.linear_encoder=nn.Linear(2*self.h_size,self.latent_size)
        self.encoder_mean = nn.Linear(self.latent_size , self.latent_size)
        self.encoder_logv = nn.Linear(self.latent_size , self.latent_size)
        
        # DECODER: Multilayer GRU, trained with teacher forcing
        # We reuse the embedding layer before passing input to rnn_decoder, hence input_size = h_size
        self.rnn_decoder = nn.GRU(input_size=self.embedding_dim, hidden_size=self.embedding_dim, num_layers=3,
                                  batch_first=True, bidirectional=False)
        
        self.dense_init=nn.Linear(self.latent_size,3*self.embedding_dim) # to initialise hidden state
        self.dense_out=nn.Linear(self.embedding_dim,self.voc_size)
        
        # PROPERTY REGRESSOR (may change to classifier with sigmoid and bce loss)
        self.MLP=nn.Sequential(
                nn.Linear(self.latent_size,32),
                nn.ReLU(),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,self.N_topics),
                nn.Softmax(dim=1))
        

    def encode(self, x, x_true_len):
        """ 
        Encodes a one_hot sequence to a mean and variance of shape latent_size
        Input: x, (batch_size * seq_length * voc_size)
        Out: mean and logv, (batch_size * latent_size ) 
        
        """
        x = self.emb_layer(x)
        
        packed = pack_padded_sequence(x, x_true_len.cpu().numpy(), batch_first=True)
        # Pass to 2directional rnn
        bi_output, bi_hidden = self.birnn(packed)
        
        # Keep last hidden state of both layers and pass to dense layer
        bi_hidden = bi_hidden.view(x.shape[0],-1)
        bi_hidden=F.selu(self.linear_encoder(bi_hidden))

        mean = self.encoder_mean(bi_hidden)
        logv = self.encoder_logv(bi_hidden)
        return mean, logv

    def sample(self, mean, logv, train=True):
        """ Samples a vector according to the latent vector mean and variance """
        if train:
            sigma = torch.exp(.5 * logv)
            return mean + torch.randn_like(mean) * sigma
        else:
            return mean

    def decode(self, z, x_true=None):
        """
            Unrolls decoder RNN to generate a batch of sequences, using teacher forcing
            Args:
                z: (batch_size * latent_shape) : a sampled vector in latent space
                x_true: (batch_size * sequence_length) a batch of sequence (word indices)
            Outputs:
                gen_seq : (batch_size * seq_length * voc_size) a batch of generated sequences
                
        """
        batch_size, seq_len =x_true.shape
        
        # Init hidden with z sampled in latent space
        h0=self.dense_init(z).view(3,batch_size, self.embedding_dim)
        
        start_token = 2 # index of SOS token in embeddings table
        start=np.ones((batch_size,1))*start_token
        
        x_offset = torch.cat((torch.tensor(start,dtype=torch.long).to(self.device), x_true[:,:-1]),dim=1 )
        x_offset = self.emb_layer(x_offset)
        # pass to GRU with teacher forcing
        out, h = self.rnn_decoder(x_offset, h0)
        out=self.dense_out(out) # Go from hidden size to voc size 
        #gen_seq = F.softmax(out, dim=1) # Shape N, voc_size
        return out
    
    def get_properties(self,z):
        """ Forward pass of the latent vector to generate molecular properties"""
        return self.MLP(z)


    def forward(self, x, x_true_len , train=True):
        """encodes x, samples in the distribution and decodes"""

        mean, logv = self.encode(x, x_true_len)
        z = self.sample(mean, logv, train=train)
        gen_seq =self.decode(z, x)
        
        properties=self.get_properties(z)
            
        return gen_seq, mean, logv, properties

def Loss(out, x_indices, mu, logvar, y=None, pred_label=None, kappa=1.0):
    """ 
    Loss function, 3 components;
        - reconstruction 
        - KL Divergence
        - mean squared error for properties prediction
    
    """
    rec = F.cross_entropy(out.transpose(2,1), x_indices, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    error= F.cross_entropy(pred_label, y.view(-1), reduction="sum")
    
    total_loss=rec + kappa*KLD + error
        
    return total_loss, rec, KLD, error # returns 4 values
     
    
def set_kappa(epoch, batch_idx,N_batches):
    """ Sets the kappa coefficient for the KLD in objective function.
        cyclical scheduling after epoch 5. 
    """
    if(epoch<=30):
        return 0
    else:
        kappa = min(2*batch_idx/N_batches,1)
        return kappa
    
def load_my_state_dict(model, state_dict):
    """
    Function to load pretrained weights for encoder and decoder, and 
    use them in a multitask model with additional MLP.
    
    Loads only weights for encoder / decoder, not the MLP ones.
    """
    own_state = model.state_dict() # The new model 
    for name, param in state_dict.items(): # The parameters saved from last time
        # DO NOT load params not present in the new model, or params of the affinity predictor. 
        if (name not in own_state):
             continue
        #if('aff_net' in name):
        #continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def lr_decay(optim, n_epoch , lr_schedule):
    lr = lr_schedule['initial_lr'] * (lr_schedule['decay_factor'] ** (n_epoch // lr_schedule['decay_freq']))
    for param_group in optim.param_groups:
        param_group['lr'] = lr
