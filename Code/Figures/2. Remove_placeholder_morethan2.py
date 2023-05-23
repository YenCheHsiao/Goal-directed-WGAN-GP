# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:51:30 2023

@author: ych22001
"""

import pandas as pd

#%% Set file directory
data_file = 'D:/PhD thesis/GAN/Github/'

#%%
method = 'GAN_RL_old'
# method = 'GAN_RL_imm_select_LG_mean_only'
ep = 'epoch1000'

num_epochs = 1000 # generator trained epoch
batch_size = 10000 # number of generated peptides
# Database: http://biopharm.zju.edu.cn/tsnadb/
#Bladder = pd.read_csv('D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '.txt',sep='\t')
Bladder = pd.read_csv(data_file + 'Data/In paper/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt',sep='\t')
# https://pandas.pydata.org/docs/reference/api/pandas.Series.str.count.html
Bladder_count = Bladder['peptide'].str.count('-')
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
Bladder_rmv = Bladder.loc[Bladder_count<2]
Bladder_out = Bladder_rmv
Bladder_out['peptide'] = Bladder_rmv['peptide'].str.replace('-', '')

import os
outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
print("outdir is {}".format(outdir))
outname = 'deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_rmv.txt'
Bladder_out.to_csv(os.path.join(outdir,outname),sep='\t',index=None)

# https://pandas.pydata.org/docs/reference/api/pandas.Series.str.len.html
#Bladder_length = Bladder['peptide'].str.len()
# https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
#Bladder_out = Bladder.loc[Bladder['peptide'].str.len()>=9]



