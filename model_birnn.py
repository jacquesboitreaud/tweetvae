# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

Required parameters:
    - MAX_LEN (max length of sequence)
    - device to run
    - N_properties (topic, polarity, etc...) any additionnal topic we would like to predict from the latent representation

TODO: adapt from model working on sequence representations of molecules 
(mostly input type and dataloading aspects to change, the rest probably remains the same )



"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence

from utils import Variable, BeamSearchNode


from queue import PriorityQueue


class tweetVAE(nn.Module):
    def __init__(self, **kwargs ):
        super(MolecularVAE, self).__init__()
        
        self.kappa=1
        self.voc_size=35
        self.max_len=kwargs['MAX_LEN']
        
        self.latent_size= 50
        self.h_size=400
        self.device=kwargs['device']
        
        
        # For multitask properties prediction
        self.N_properties=kwargs['N_properties']
        self.N_targets = kwargs['N_targets']
        
        
        # ENCODER
        self.birnn = torch.nn.GRU(input_size=35, hidden_size=400, num_layers=1,
                                  batch_first=True, bidirectional=True)
        
        # Latent mean and variance
        self.linear_encoder=nn.Linear(800,self.latent_size)
        self.encoder_mean = nn.Linear(self.latent_size , self.latent_size)
        self.encoder_logv = nn.Linear(self.latent_size , self.latent_size)
        
        # DECODER: Multilayer GRU, trained with teacher forcing
        self.rnn_decoder = nn.GRU(input_size=35, hidden_size=400, num_layers=3,
                                  batch_first=True, bidirectional=False)
        
        self.dense_init=nn.Linear(self.latent_size,3*self.h_size) # to initialise hidden state
        self.dense_out=nn.Linear(400,35)
        
        # PROPERTY REGRESSOR (may change to classifier with sigmoid and bce loss)
        self.MLP=nn.Sequential(
                nn.Linear(self.latent_size,64),
                nn.ReLU(),
                nn.Linear(64,self.N_properties))
        

    def encode(self, x, x_true_len):
        """ 
        Encodes a one_hot sequence to a mean and variance of shape latent_size
        Input: x, (batch_size * seq_length * voc_size)
        Out: mean and logv, (batch_size * latent_size ) 
        
        """
        # Pass to 2directional rnn
        packed = pack_padded_sequence(x, x_true_len, batch_first=True)
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
                x_true: (batch_size * sequence_length * voc_size) a batch of sequences
            Outputs:
                gen_seq : (batch_size * seq_length * voc_size) a batch of generated sequences
                
        """
        batch_size=z.shape[0]
        
        # Init hidden with z sampled in latent space
        h0=self.dense_init(z).view(3,batch_size, self.h_size)
        
        x_offset = torch.cat((torch.zeros(batch_size,1,self.voc_size).to(self.device), x_true[:,:-1,:]),dim=1 )
        # pass to GRU with teacher forcing
        out, h = self.rnn_decoder(x_offset, h0)
        out=self.dense_out(out) # Go from hidden size to voc size 
        
        gen_seq = F.softmax(out, dim=1) # Shape N, voc_size
                
        return gen_seq
    
    def decode_beam(self, z, k=3):
        """
        Decodes a batch, sequence after sequence, using beam search of width k (more precise decoding, robust to errors)
        Output: 
            a list of lists of k best sequences for each molecule.
        TODO : rewrite with pytorch GRU implementation 
        """
        N = z.shape[0]
        sequences = []
        for n in range(N):
            print("decoding molecule n° ",n)
            # Initialize rnn states and input
            z_1mol=z[n].view(1,self.latent_size) # Reshape as a batch of size 1
            start_token = self.rnn_in(z_1mol).view(1,1,self.voc_size).to(self.device)
            rnn_in = start_token
            h = self.rnn.init_h(z_1mol)
            topk = [BeamSearchNode(h,rnn_in, 0, [] )]
            
            for step in range(self.max_len):
                next_nodes=PriorityQueue()
                for candidate in topk: # for each candidate sequence (among k)
                    score = candidate.score
                    seq=candidate.sequence
                    # pass into decoder
                    out, new_h = self.rnn(candidate.rnn_in, candidate.h) 
                    probas = F.softmax(out, dim=1) # Shape N, voc_size
                    for c in range(self.voc_size):
                        new_seq=seq+[c]
                        rnn_in=torch.zeros((1,35))
                        rnn_in[0,c]=1
                        s= score-probas[0,c]
                        next_nodes.put(( s.item(), BeamSearchNode(new_h, rnn_in.to(self.device),s.item(), new_seq)) )
                topk=[]
                for k_ in range(k):
                    # get top k for next timestep !
                    score, node=next_nodes.get()
                    topk.append(node)
                    #print("top sequence for next step :", node.sequence)
                    
            sequences.append([n.sequence for n in topk]) # list of lists 
        return sequences
    
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

def Loss(out, x, mu, logvar, y=None, pred_properties=None, kappa=1.0):
    """ 
    Loss function for VAE with reconstruction loss, divergence 
    
    """
    BCE = F.binary_cross_entropy(out, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    
    error= F.mse_loss(pred_properties, y, reduction="sum")
    total_loss=BCE + kappa*KLD + error
        
    return total_loss, BCE, KLD, error # returns 4 values
     
    
def set_kappa(epoch, batch_idx,N_batches):
    """ Sets the kappa coefficient for the KLD in objective function.
        cyclical scheduling after epoch 5. 
    """
    if(epoch<=5):
        return 0
    else:
        kappa = min(2*batch_idx/N_batches,1)
        return kappa
