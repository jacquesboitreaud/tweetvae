# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:01:48 2019

@author: jacqu

Pretrains the plain VAE to reconstruct smiles.
CAN 2 CAN setting 
Bidirectional RNN as encoder 

=> For debugging, set L to a small value. Will prune the train set and run an epoch fast.  
"""

import os
import numpy as np
import time
from random import shuffle

import torch
import torch.utils.data
from torch import nn, optim

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler

from model_birnn import tweetVAE, set_kappa, Loss

from numpy.linalg import norm

# ============= Model Params =================================================

# File to save the model's weights
SAVE_FILENAME='./saved_model_w/c2c.pth'
LOGS='../data/logs_c2c.npy'
# To load model 
SAVED_MODEL_PATH ='./saved_model_w/c2c.pth'
LOAD_MODEL=False # set to true to load pretrained model
SAVE_MODEL=True

PRETRAIN = True # Train under plain VAE loss, without affinities 

# Model params dict
model_params={'MAX_LEN': 150,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'N_properties':10,
        'N_targets':102} 
parallel=False

# Training params
epochs=10
batch_size=128
lr_schedule={'initial_lr':3e-4,
             'decay_freq':5,
             'decay_factor':0.9}
# Prune training data to make a fast try
L=None


# Sequence to rewrite and adapt : dataloading 

train_csv="../data/DUD_full.csv"

train_set = ligandsDataset_c2c(train_csv, limit=L, properties = prop_names)
test_set = ligandsDataset_c2c(train_csv, limit = 128, properties = prop_names)


U_indices = np.arange(len(train_set))
loader =torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(U_indices))

    

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
print('Train set contains {} samples, test set contains {}'.format(len(train_set),len(test_set)))

# ====================== Instantiate model  =================================

model = tweetVAE(**model_params ).to(model_params['device']).float()

if (parallel): #torch.cuda.device_count() > 1 and
  print("Start training using ", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)

#Print model summary
print(model)
map = ('cpu' if model_params['device'] == 'cpu' else None)


start = time.time()
torch.manual_seed(18)
optimizer = optim.Adam(model.parameters(),lr=lr_schedule['initial_lr'], 
                       weight_decay=1e-5)
# kappa annealing 
kappa=0
N_batches = int(len(train_set)/batch_size)


def on_epoch_end(TEST_SCORED=False, save_pred_file=None):
    """ Setting for unlabeled test set, without affinity prediction """
    # Performance tests at the end of each batch.
    
    with torch.no_grad():
        # test set pass
        l_tot, rec_tot, mse_tot, aff_tot=0,0,0,0

        for batch_idx, data in enumerate(test_loader):
        
            truesmiles = data[0].to(model_params['device'])
            seq_lengths = data[1].to(model_params['device'])
            p_target= data[2].to(model_params['device'])
            
            # Sort 
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            truesmiles = truesmiles[perm_idx]
            
            
            #=========== forward ==========================
            recon_batch, mu, logvar, prop= model(truesmiles, seq_lengths)
            t_loss, rec, div, mse = Loss(recon_batch, truesmiles, mu, logvar,
                                                         y=p_target, pred_properties= prop,
                                                         kappa = kappa)
                
           
                
            l_tot += t_loss.item()
            rec_tot += rec.item()
            mse_tot += mse.item()
            
            # Print some monitoring stats for the first batch:
            if(batch_idx==0):
                
                # Monitor loss at a character level
                loss_per_char = rec.item()/(150*recon_batch.shape[0])
                print("Test cross-entropy loss per char: ", loss_per_char)
            
                # Print mu and logvar mean for one batch
                mu_, logvar_=mu.cpu().numpy(), logvar.cpu().numpy()
                print('mean norm: ',np.mean(norm(mu_,axis=0)))
                print('logvar norm: ',np.mean(norm(logvar_,axis=0)))
            
        return l_tot, rec_tot, mse_tot
        
    
        
def lr_decay(optim, n_epoch , lr_schedule):
    lr = lr_schedule['initial_lr'] * (lr_schedule['decay_factor'] ** (epoch // lr_schedule['decay_freq']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch, pre_train=False):
    # trains the model for one epoch. Returns the loss per item for this epoch.
    model.train()
    l_tot, rec_tot, kl_tot, mse_tot, aff_tot=0,0,0,0,0
    
    
    print ("\n")
    print("starting epoch {}".format(epoch))
    print("\n")

    # Unlabeled loader : No aff loss !!!!!!!!!!!!!!
    for batch_idx, data in enumerate(U_loader):
        
        kappa=set_kappa(epoch, batch_idx,N_batches)
        
        truesmiles = data[0].to(model_params['device'])
        seq_lengths = data[1].to(model_params['device'])
        p_target= data[2].to(model_params['device'])
        
        # Sort 
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        truesmiles = truesmiles[perm_idx]
        
        #=========== forward ==========================
        recon_batch, mu, logvar, prop = model(truesmiles, seq_lengths)
        tr_loss, rec, div, mse = Loss(recon_batch,truesmiles, mu, logvar,
                                                     y=p_target, pred_properties= prop,
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
    
    return l_tot, rec_tot, kl_tot, mse_tot, aff_tot



# ============== Train and test ==============

if (__name__=="__main__"):
    
    #  ========== load pretrained model or restart from scratch ? =============
    if(LOAD_MODEL):
        print("loading network coeffs")
        load_my_state_dict(model, torch.load(SAVED_MODEL_PATH, map_location=map))
    # =========================================================================
    logs={'train_l':[],
           'train_rec':[],
           'test_rec':[],
           'train_div':[],
           'train_mse':[],
           'test_mse':[],
           'test_l':[]}
    
    
    for epoch in range(1, epochs + 1):
        # train step
        t0=time.time()
        epoch_loss, rec_tot, kl_tot, mse_tot, aff_tot = train(epoch, PRETRAIN)
        t1=time.time()
        delta=t1-t0
        print('epoch [{}/{}], loss per item: {:.2f}, time elapsed: {:.1f}'.format(epoch, 
              epochs, epoch_loss, delta))
        
        # test step
        test_loss, test_rec, test_mse, test_aff_loss = on_epoch_end(TEST_SCORED=False, 
                                                                    save_pred_file='../data/sample_pred.csv')
        
        # Append logs
        logs['train_l'].append(epoch_loss)
        logs['test_l'].append(test_loss)
        
        logs['train_rec'].append(rec_tot)
        logs['test_rec'].append(test_rec)
        
        logs['train_div'].append(kl_tot)
        
        logs['train_mse'].append(mse_tot)
        logs['test_mse'].append(test_mse)
        
        # Save dict
        np.save(LOGS, logs)
        
        if (SAVE_MODEL): # Save model every epoch
            torch.save( model.state_dict(), SAVE_FILENAME)
            print("model saved")
            best_loss=epoch_loss
