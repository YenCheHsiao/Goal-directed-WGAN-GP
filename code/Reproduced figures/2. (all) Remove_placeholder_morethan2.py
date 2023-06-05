# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 13:41:42 2023

@author: xiaoyenche
"""

import pandas as pd

#%% Set file directory
data_file = '../../'

#%%
# method = 'WGAN-GP'
method = 'Goal-directed_WGAN-GP'

# For 0,20,40,60,80,100 epoch
# ep = 'epoch0'
# num_epochs = 0 # generator trained epoch
# batch_size = 1024 # number of generated peptides

# For 1000 epoch
ep = 'epoch1000'
num_epochs = 1000 # generator trained epoch
batch_size = 100000 # number of generated peptides

#%%
method = 'Goal-directed_WGAN-GP'
ep = ['epoch0','epoch20','epoch40','epoch60','epoch80','epoch100','epoch1000']
num_epochs = [0,20,40,60,80,100,1000]
batch_size = [1024,1024,1024,1024,1024,1024,10000]
for j in range(len(ep)):
    #%%
    # Database: http://biopharm.zju.edu.cn/tsnadb/
    #Bladder = pd.read_csv('D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '.txt',sep='\t')
    Bladder = pd.read_csv(data_file + 'results/' + method + '/' + ep[j] + '/deepimmuno-GANRL-bladder-epoch' + str(num_epochs[j]) + '-batch' + str(batch_size[j]) + '.txt',sep='\t')
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.count.html
    Bladder_count = Bladder['peptide'].str.count('-')
    # https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
    Bladder_rmv = Bladder.loc[Bladder_count<2]
    Bladder_out = Bladder_rmv
    Bladder_out['peptide'] = Bladder_rmv['peptide'].str.replace('-', '')
    
    import os
    outdir = data_file + 'results/' + method + '/' + ep[j] + '/'
    print("outdir is {}".format(outdir))
    outname = 'deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_rmv.txt'
    Bladder_out.to_csv(os.path.join(outdir,outname),sep='\t',index=None)
    
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.len.html
    #Bladder_length = Bladder['peptide'].str.len()
    # https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
    #Bladder_out = Bladder.loc[Bladder['peptide'].str.len()>=9]

#%%
method = 'WGAN-GP'

# For 0,20,40,60,80,100 epoch
# ep = 'epoch0'
# num_epochs = 0 # generator trained epoch
# batch_size = 1024 # number of generated peptides

# For 1000 epoch
ep = 'epoch1000'
num_epochs = 1000 # generator trained epoch
batch_size = 10000 # number of generated peptides

#%%
# Database: http://biopharm.zju.edu.cn/tsnadb/
#Bladder = pd.read_csv('D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '.txt',sep='\t')
Bladder = pd.read_csv(data_file + 'results/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt',sep='\t')
# https://pandas.pydata.org/docs/reference/api/pandas.Series.str.count.html
Bladder_count = Bladder['peptide'].str.count('-')
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
Bladder_rmv = Bladder.loc[Bladder_count<2]
Bladder_out = Bladder_rmv
Bladder_out['peptide'] = Bladder_rmv['peptide'].str.replace('-', '')

import os
outdir = data_file + 'results/' + method + '/' + ep + '/'
print("outdir is {}".format(outdir))
outname = 'deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_rmv.txt'
Bladder_out.to_csv(os.path.join(outdir,outname),sep='\t',index=None)

# https://pandas.pydata.org/docs/reference/api/pandas.Series.str.len.html
#Bladder_length = Bladder['peptide'].str.len()
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
#Bladder_out = Bladder.loc[Bladder['peptide'].str.len()>=9]


