# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:51:30 2023

@author: ych22001
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



