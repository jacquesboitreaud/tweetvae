# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Trains sequence-to-sequence VAE to learn tweet embedding  

"""
import sys
import pickle
import numpy as np
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
    n_epochs = 60 # epochs to train
    batch_size = 64
    # File to save the model's weights
    SAVE_FILENAME='./saved_model_w/first_try.pth'
    LOGS='../data/logs_first_try.npy'
    # To load model 
    SAVED_MODEL_PATH ='./saved_model_w/first_try.pth'
    LOAD_MODEL=False # set to true to load pretrained model
    SAVE_MODEL=True
    
    #Load train set and test set
    loaders = Loader(num_workers=4, batch_size=batch_size, clean= True, max_n=200000)
    train_loader, _, test_loader = loaders.get_data()
    # Save vocabulary for later (evaluation):
    pickle.dump(loaders.dataset.words_to_ids,open("./saved_model_w/vocabulary.pickle","wb"))
    
    #Model & hparams
    
    
    model_params={'MAX_LEN': loaders.dataset.max_words,
                  'vocab_size': loaders.dataset.voc_len,
                  'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                  'N_properties':1,
                  'N_topics':5} 
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
    lr_schedule={'initial_lr':5e-4,
             'decay_freq':4,
             'decay_factor':0.8}
    optimizer = optim.SGD(model.parameters(), lr= lr_schedule['initial_lr'])
    #optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)
    
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
            kappa=set_kappa(epoch, batch_idx,len(train_loader))
            #kappa=0
            
            # Get data to GPU
            true_idces = data[0]
            seq_lengths = data[1] #1D int CPU tensor
            #print(len(seq_lengths), true_idces.shape)
            
            l_target= data[2].to(model_params['device']).view(-1,1)
            
            seq_tensor = torch.zeros((batch_size,model.max_len)).long().cuda()
            for idx, seq in enumerate(true_idces):
                seq_tensor[idx] = torch.LongTensor(seq)
            
            # Sort 
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            true_idces,l_target = true_idces[perm_idx], l_target[perm_idx]
            
            #print(seq_tensor.shape)
            #print(seq_lengths.shape)
            
            #=========== forward ==========================
            recon_batch, mu, logvar, label = model(seq_tensor, seq_lengths)
            tr_loss, rec, div, mse = Loss(recon_batch,true_idces.to(model_params['device']), mu, logvar,
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
        l_tot, rec_tot, div_tot, mse_tot = 0,0,0,0
        with torch.no_grad():
            # test set pass
        
            for batch_idx, data in enumerate(test_loader):
            
                true_idces = data[0].to(model_params['device'])
                seq_lengths = data[1]
                l_target= data[2].to(model_params['device']).view(-1,1)
                
                # Sort 
                seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
                true_idces = true_idces[perm_idx]
                
                #=========== forward ==========================
                recon_batch, mu, logvar, label= model(true_idces, seq_lengths)
                
                t_loss, rec, div, mse = Loss(recon_batch, true_idces.to(model_params['device']), mu, logvar,
                                                             y=l_target, pred_label= label,
                                                             kappa = 0)
                    
               
                    
                l_tot += t_loss.item()
                rec_tot += rec.item()
                div_tot += div.item()
                mse_tot += mse.item()
                
                # Print some monitoring stats for the first batch of test subset:
                if(batch_idx==0):
                    # Monitor loss at word level
                    loss_per_timestep = rec.item()/(150*recon_batch.shape[0])
                    print("Test cross-entropy loss per word : ", loss_per_timestep)
                    
                    # Print some decoded tweets: 
                    recon_batch = recon_batch.cpu().numpy()
                    N = recon_batch.shape[0]
                    for k in range(N):
                        prev_word, tweet=' ', ' '
                        timestep = 0
                        while(prev_word!='EOS'):
                            prev_word=loaders.dataset.ids_to_words(np.argmax(recon_batch[k,timestep]))
                            tweet += prev_word
                            tweet+=' '
                        print(tweet)
                        
                        
            # At the end of the test set pass : 
            logs['test_l'].append(l_tot)
            logs['test_mse'].append(mse_tot)
            logs['test_rec'].append(rec_tot)
            logs['test_div'].append(div_tot)
            
            if (SAVE_MODEL): # Save model every epoch
                torch.save( model.state_dict(), SAVE_FILENAME)
                print(f"model saved to {SAVE_FILENAME}")
                best_loss=l_tot
        