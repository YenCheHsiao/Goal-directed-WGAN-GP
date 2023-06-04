# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:39:50 2023

@author: ych22001
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Set file directory
data_file = '../../'

#%%
num_epochs = 400
num_epochs = 1000
num_epochs_1000 = 1000

# Select only LG mean method
# 1-10 GAN+CNN (scored substract)
method = 'Goal-directed_WGAN-GP'
input_dir1 = data_file + 'results/' + method + '/' + 'epoch' + str(num_epochs_1000)
score1 = np.load(input_dir1+'/G_score.npy')
temp = 0
acu_score1 = []
for i in range(0, num_epochs, 1): # range(start, stop, step)
    temp = temp + score1[i]
    acu_score1.append((temp/(i+1)))

# No predictor
# 1. GAN+CNN (no label)
method = 'WGAN-GP'
input_dir2 = data_file + 'results/' + method + '/' + 'epoch' + str(num_epochs_1000)
score2 = np.load(input_dir2+'/G_score.npy')
temp = 0
acu_score2 = []
for i in range(0, num_epochs, 1): # range(start, stop, step)
    temp = temp + score2[i]
    acu_score2.append((temp/(i+1)))
    
plt.figure(figsize=(6.4*1.2, 4.8*1.2), dpi=1000)
plt.plot(np.arange(num_epochs), score2[:num_epochs], c='#1f77b4',alpha=0.75)   
plt.plot(np.arange(num_epochs), score1[:num_epochs], c='#ff7f0e',alpha=0.75) 
plt.plot(np.arange(num_epochs), acu_score2[:num_epochs], c='#9932CC', linestyle='dotted', linewidth=4)
plt.plot(np.arange(num_epochs), acu_score1[:num_epochs], c='#800000', linestyle='dashed', linewidth=4)
plt.legend(['WGAN-GP','Goal-directed WGAN-GP'],fontsize = 15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Training epoch',fontsize = 20, weight='bold')
plt.ylabel('Averaged immunogenicity score',fontsize = 20, weight='bold')
plt.title('Comparison of different designs',fontsize = 20, weight='bold')
plt.grid()
plt.savefig(data_file + "results/(New) Learning curve.png", bbox_inches='tight')
plt.show()

