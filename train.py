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
    n_epochs = 2 # epochs to train
    batch_size = 64
    # File to save the model's weights
    SAVE_FILENAME='./saved_model_w/c2c.pth'
    LOGS='../data/logs_c2c.npy'
    # To load model 
    SAVED_MODEL_PATH ='./saved_model_w/c2c.pth'
    LOAD_MODEL=False # set to true to load pretrained model
    SAVE_MODEL=True
    
    #Load train set and test set
    loaders = Loader(num_workers=1, batch_size=batch_size, clean= False, max_n=4000)
    train_loader, _, test_loader = loaders.get_data()
    
    #Model & hparams
    
    
    model_params={'MAX_LEN': loaders.dataset.max_words,
                  'vocab_size': len(loaders.dataset.vocab),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'N_properties':1} 
    model = tweetVAE(**model_params ).to(model_params['device']).float()
    
    if(LOAD_MODEL):
        print("loading network coeffs")
        load_my_state_dict(model, torch.load(SAVED_MODEL_PATH, map_location=map))
    
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
    
    # Logs dict 
    logs={'train_l':[],
           'train_rec':[],
           'test_rec':[],
           'train_div':[],
           'test_div':[],
           'train_mse':[],
           'test_mse':[],
           'test_l':[]
           }
    
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
            true_idces = data[0].to(model_params['device'])
            seq_lengths = data[1] #1D int CPU tensor
            l_target= data[2].to(model_params['device']).view(-1,1)
            
            seq_tensor = torch.zeros((batch_size, model.max_len)).long().cuda()
            for idx, (seq, seqlen) in enumerate(zip(true_idces, seq_lengths)):
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            
            # Sort 
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            true_idces,l_target = true_idces[perm_idx], l_target[perm_idx]
            
            #=========== forward ==========================
            recon_batch, mu, logvar, label = model(true_idces, seq_lengths)
            tr_loss, rec, div, mse = Loss(recon_batch,true_idces, mu, logvar,
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
    
        
        # At the end of each training epoch:
        #Decay learning rate: 
        lr_decay(optimizer,epoch, lr_schedule)
        
        # Append logs
        logs['train_l'].append(l_tot)
        logs['train_rec'].append(rec_tot)
        logs['train_div'].append(kl_tot)
        logs['train_mse'].append(mse_tot)
    
    
    # Validation pass
    model.eval()
    l_tot, rec_tot, div_tot, mse_tot = 0
    with torch.no_grad():
        # test set pass
    
        for batch_idx, data in enumerate(test_loader):
        
            true_idces = data[0].to(model_params['device'])
            seq_lengths = data[1]
            l_target= data[2].to(model_params['device']).view(-1,1)
            
            # Sort 
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            true_seqs = true_seqs[perm_idx]
            
            #=========== forward ==========================
            recon_batch, mu, logvar, label= model(true_idces, seq_lengths)
            
            t_loss, rec, div, mse = Loss(recon_batch, true_seqs, mu, logvar,
                                                         y=l_target, pred_label= label,
                                                         kappa = 0)
                
           
                
            l_tot += t_loss.item()
            rec_tot += rec.item()
            div_tot += div.item()
            mse_tot += mse.item()
            
            # Print some monitoring stats for the first batch of test subset:
            if(batch_idx==0):
                # Monitor loss at a character level
                loss_per_timestep = rec.item()/(150*recon_batch.shape[0])
                print("Test cross-entropy loss per word : ", loss_per_timestep)
        
        # At the end of the test set pass : 
        logs['test_l'].append(l_tot)
        logs['test_mse'].append(mse_tot)
        logs['test_rec'].append(rec_tot)
        logs['test_div'].append(div_tot)
        