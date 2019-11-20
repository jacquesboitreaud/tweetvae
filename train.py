# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Trains sequence-to-sequence VAE to learn tweet embedding  

"""
import sys
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

from model_birnn import tweetVAE, Loss, set_kappa, load_my_state_dict, lr_decay
from tweetDataset import tweetDataset, Loader
from utils import *

if __name__ == '__main__': 

    # config
    n_epochs = 20 # epochs to train
    batch_size = 64
    
    #Load train set and test set
    loaders = Loader(num_workers=1, batch_size=batch_size, clean= False, max_n=40000)
    train_loader, _, test_loader = loaders.get_data()
    
    #Model & hparams
    
    
    model_params={'MAX_LEN': loaders.dataset.max_words,
                  'vocab_size': len(loaders.dataset.vocab),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'N_properties':1} 
    model = tweetVAE(**model_params ).to(model_params['device']).float()
    
    parallel=False
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Start training using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    #Print model summary
    print(model)
    map = ('cpu' if model_params['device'] == 'cpu' else None)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)
    lr_schedule={'initial_lr':5e-4,
             'decay_freq':4,
             'decay_factor':0.8}
    
    #Train & test
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        
        l_tot, rec_tot, kl_tot, mse_tot=0,0,0,0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            
            # VAE loss annealing 
            #kappa=set_kappa(epoch, batch_idx,N_batches)
            kappa=1
            
            # Get data to GPU
            true_seqs = data[0].to(model_params['device'])
            seq_lengths = data[1] #1D int CPU tensor
            l_target= data[2].to(model_params['device']).view(-1,1)
            
            # Sort 
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            true_seqs,l_target = true_seqs[perm_idx], l_target[perm_idx]
            
            #=========== forward ==========================
            recon_batch, mu, logvar, label = model(true_seqs, seq_lengths)
            tr_loss, rec, div, mse = Loss(recon_batch,true_seqs, mu, logvar,
                                                         y=l_target, pred_label= label,
                                                         kappa = kappa)
                
            # =========== backward =========================
            optimizer.zero_grad()
            tr_loss.backward()
            clip.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
            
            # ===========logs and monitoring================
            if batch_idx % 100 == 0:
                # log
                print('ep {}, batch {}, rec : {:.2f}, div: {:.2f}, mse: {:.2f} '.format(epoch, 
                      batch_idx,
                      rec.item(), div.item(),mse.item() ))
                
                
            l_tot += tr_loss.item()
            rec_tot += rec.item()
            kl_tot += div.item()
            mse_tot += mse.item()
    
        
        # Decay learning rate: 
        lr_decay(optimizer,epoch, lr_schedule)
    
    
    # Validation pass
    model.eval()
    t_loss = 0
    with torch.no_grad():
        # test set pass
        l_tot, rec_tot, mse_tot =0,0,0
    
        for batch_idx, data in enumerate(test_loader):
        
            true_seqs = data[0].to(model_params['device'])
            seq_lengths = data[1]
            l_target= data[2].to(model_params['device']).view(-1,1)
            
            # Sort 
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            true_seqs = true_seqs[perm_idx]
            
            #=========== forward ==========================
            recon_batch, mu, logvar, label= model(true_seqs, seq_lengths)
            t_loss, rec, div, mse = Loss(recon_batch, true_seqs, mu, logvar,
                                                         y=l_target, pred_label= label,
                                                         kappa = 0)
                
           
                
            l_tot += t_loss.item()
            rec_tot += rec.item()
            mse_tot += mse.item()
            
            # Print some monitoring stats for the first batch of test subset:
            if(batch_idx==0):
                # Monitor loss at a character level
                loss_per_timestep = rec.item()/(150*recon_batch.shape[0])
                print("Test cross-entropy loss per word : ", loss_per_timestep)
        