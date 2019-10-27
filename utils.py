# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:14:11 2019

@author: jacqu

Utils functions for preprocessing
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen, Descriptors, rdMolDescriptors, GraphDescriptors
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch

from torch.utils.data import Dataset

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if (torch.cuda.is_available()) :
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


class Preprocessor():
    
    def __init__(self, smiles_length ,char_file="../data/zinc_chars.json"
                 , padding='right'):
        super(Preprocessor, self).__init__()
        
        self.MAX_LEN=smiles_length
        self.PADDING=padding
        
        self.char_list = json.load(open(char_file))
        self.n_chars = len(self.char_list)
        print("SMILES dictionary contains {} characters".format(self.n_chars))

        # Build dict
        self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))
        
        
    def filter_set(self,df_ok):
        """
        filters a pandas dataframe with cols len, smiles and returns 
        the cleaned dataframe
        """
        df_ok.reset_index()
        # remove smiles containing bad chars by iterating on rows
        for i, row in df_ok.iterrows():
            if(('.' in row['smiles']) or ('Si' in row['smiles'])):
                df_ok=df_ok.drop(i,0)
                
        return df_ok
        
    def split(self,df):
        """
        Input: a filtered dataframe with cols is_ligand and smiles
        Output: two dataframes, mutually exclusive 
        """
        test = df.sample(1000)
        train = df.drop(test.index)

        return train, test

        
    
    def one_hot(self,smiles):
        """
        Input: a list of smiles strings of length MAX_LEN (padded)
        Output: a 3D numpy array, ( n_smiles * MAX_LEN * voc_size )
        """
        N=len(smiles)
        X=np.zeros((N,self.MAX_LEN, self.n_chars))
        for i, smile in enumerate(smiles): # for each molecule in the list
            for t, char in enumerate(smile): # for each char in molecule
                X[i, t, self.char_to_index[char]] = 1
        
        return X

         
    def to_smiles(self,arr):
        """
        Input: a np array of one (2-dim) or several (3-dim) onehot encoded smiles
        Output: a smile or a list of smiles
        """
        if (arr.ndim==2):
            length, n_chars=arr.shape
            smile=''
            for position in range(length):
                j=np.argmax(arr[position])
                smile+=str(self.index_to_char[j])
            return smile
        
        else: # multiple smiles in array
            n_mol, length, n_chars=arr.shape
            smiles=[]
            for i_mol in range(n_mol):
                smi=''
                for position in range(length):
                    j=np.argmax(arr[i_mol,position])
                    smi+=str(self.index_to_char[j])
                smiles.append(smi)
            return smiles

    def isValid(self,smi):
        """
        Input: a smiles string
        Output: Bool , says whether it is a valid smiles
        """
        mol=Chem.MolFromSmiles(smi) 
        # With sanitize keword arg, it checks whether the smiles is sintaxically valid
        # Without, it checks if it is reasonable in terms of chemistry
        
        if(mol==None):
            return False
        return True
        
    def index_to_smiles(self,arr):
        """ 1D array as input, only one smiles """
        smile=''
        n_chars=arr.shape[0]
        for i in range(n_chars):
            smile+=self.index_to_char[arr[i]]
        return smile.rstrip(' ')
    
    def indices_to_smiles(self,liste):
        smile=''
        for i in liste:
            smile+=self.index_to_char[i]
        return smile.rstrip(' ')
    
    
    def compute_10props(self, smiles):
        """
        Input : 
            either a sole smiles or a list of smiles 
        """
        if(type(smiles)==str):
            m=Chem.MolFromSmiles(smiles)
            arr = np.zeros(10)
            # Just to remember in which order the properties are computed ()
            prop_names=['QED','logP','molWt','maxCharge','minCharge','valence','TPSA','HBA','HBD','jIndex']
            arr[0]=QED.default(m)
            arr[1]=Crippen.MolLogP(m)
            arr[2]=Descriptors.MolWt(m)
            arr[3]=Descriptors.MaxPartialCharge(m)
            arr[4]=Descriptors.MinPartialCharge(m)
            arr[5]=Descriptors.NumValenceElectrons(m)
            arr[6]=rdMolDescriptors.CalcTPSA(m)
            arr[7]=rdMolDescriptors.CalcNumHBA(m)
            arr[8]=rdMolDescriptors.CalcNumHBD(m)
            arr[9]=GraphDescriptors.BalabanJ(m)
            return arr
        else : # a list of smiles is provided
            smiles=list(smiles)
            N = len(smiles)
            a = np.zeros((N,10))
            for i in range(N):
                a[i]=self.compute_10props(smiles[i])
            return a 
        
    def compute_9props(self, smiles):
        """
        Input : 
            either a sole smiles or a list of smiles 
        """
        if(type(smiles)==str):
            m=Chem.MolFromSmiles(smiles)
            arr = self.compute_10props(smiles)[1:]
            # Just to remember in which order the properties are computed ()
            prop_names=['logP','molWt','maxCharge','minCharge','valence','TPSA','HBA','HBD','jIndex']
            return arr
        else : # a list of smiles is provided
            smiles=list(smiles)
            N = len(smiles)
            a = np.zeros((N,9))
            for i in range(N):
                a[i]=self.compute_9props(smiles[i])
            return a 
    

class BeamSearchNode():
    def __init__(self, h, rnn_in, score, sequence):
        self.h=h
        self.rnn_in=rnn_in
        self.score=score
        self.sequence=sequence
        self.max_len = 60 

    def __lt__(self, other): # For x < y
        # Pour casser les cas d'égalité du score au hasard, on s'en fout un peu.
        # Eventuellement affiner en regardant les caractères de la séquence (pénaliser les cycles ?)
        return True
    
class oneHot(object):
    """ 
    
    Turns the smiles to a one hot array of shape max_len * voc_size 
    Returns tuple (one hot tensor, smiles length) for packed input handling 
    
    """
    def __init__(self, max_len, char_file="../data/zinc_chars.json"):
        self.max_len=max_len
        self.char_list = json.load(open(char_file))
        self.char_to_index= dict((c, i) for i, c in enumerate(self.char_list))
        self.n_chars=len(self.char_list)

    def __call__(self, smi):
        a=np.zeros((self.max_len,self.n_chars), dtype = np.float32)
        for t, char in enumerate(smi):
            a[t, self.char_to_index[char]] = 1
        for t in range(len(smi),self.max_len):
            a[t, self.char_to_index[' ']] = 1
            
        return torch.from_numpy(a), len(smi)


class toIndex(object):
    """ 
    Turns the smiles to a target array of shape (max_len)  
    """
    def __init__(self, max_len, char_file="../data/zinc_chars.json"):
        self.max_len=max_len
        self.char_list = json.load(open(char_file))
        self.char_to_index= dict((c, i) for i, c in enumerate(self.char_list))
        self.n_chars=len(self.char_list)

    def __call__(self, smi):
        a=np.zeros((self.max_len), dtype = np.float32)
        for t, char in enumerate(smi):
            a[t]=self.char_to_index[char]
        for t in range(len(smi),self.max_len):
            a[t] =  self.char_to_index[' ']
            
        return torch.from_numpy(a)
    

        
class ConcatDataset(Dataset):
    """ Concatenated datasets but works if both are same length / same nbr of batches"""
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2

    def __getitem__(self, i):
        return self.d1.__getitem__(i) , self.d2.__getitem__(i)

    def __len__(self):
        return min(len(self.d1), len(self.d2))
        
    
class ligandsDataset_c2c(Dataset):
    """ Feeds only one smiles (canonical), different from the translater VAE setting.
    Automatically deals with affinities"""
    
    def __init__(self, 
                 csv_path, 
                 limit=None,
                 properties=['QED','logP','molWt','maxCharge','minCharge','valence','TPSA','HBA','HBD','jIndex'],
                 char_file="../data/zinc_chars.json",
                 targets = None,
                 PRETRAIN=True):
        
        # For smiles processing : 
        self.char_list = json.load(open(char_file))
        self.n_chars = len(self.char_list)
        self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))
        self.OH = oneHot( 150 , char_file)
        self.pretrain = PRETRAIN # scores or not
        
        # Loads the csv file and builds the dataset
        if(limit!=None):
            self.df=pd.read_csv(csv_path, nrows=limit)
        else: 
            self.df=pd.read_csv(csv_path)
            
        # props and targets
        self.props=properties
        if(not self.pretrain):
            if(targets!=None):
                self.targets = targets
            else:
                self.targets = list(np.load('targets.npy'))
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # gets the item n°idx in the df and returns 
        # [0] : the smiles
        # [1] : the properties vector
        # [2] : the scores 
        
        row = self.df.iloc[idx]
        p = row[self.props].to_numpy(dtype = np.float32)
        tensor, length = self.OH(row['can'])
            
        if(not self.pretrain):
            sc = np.zeros(len(self.targets),dtype = np.float32) # Table des scores 
            
            for i in range(len(self.targets)):
                if(row[self.targets[i]]==0):
                    sc[i]=np.nan
                if( row[self.targets[i]]==-1) :
                    continue # on laisse le score à zéro 
                else:
                    sc[i]=row[str(self.targets[i])]
        
            return tensor,length, torch.from_numpy(p),torch.from_numpy(sc)
        
        else:
            return tensor, length, torch.from_numpy(p)
            
