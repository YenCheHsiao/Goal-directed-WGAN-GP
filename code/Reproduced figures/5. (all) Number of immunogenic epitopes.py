# -*- coding: utf-8 -*-
'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)
'''
"""
Created on Sat Jun  3 13:58:20 2023

@author: xiaoyenche
"""
# source code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% Set file directory
data_file = '../../'

#%%
method = 'Goal-directed_WGAN-GP'
ep = ['epoch0','epoch20','epoch40','epoch60','epoch80','epoch100']
num_epochs = [0,20,40,60,80,100]
batch_size = [1024,1024,1024,1024,1024,1024]
Imm_ep = []
for j in range(len(ep)):
    #%%
    indir = data_file + 'results/' + method + '/' + ep[j] + '/'
    print("indir is {}".format(indir))
    data_path = 'deepimmuno-GANRL-bladder_cancer-scored_epoch' + str(num_epochs[j]) + '-batch' + str(batch_size[j]) + '.txt'
    imm_prediction = pd.read_csv(indir + data_path, sep='\t')
    Imm_ep.append(sum(imm_prediction['immunogenicity']>0.5))

#%%
epochs = ("0", "20", "40", "60", "80", "100")
penguin_means = {
    'DeepImmuno$^{19}$': (414, 515, 622, 650, 659, 679),
    'Our Design': (Imm_ep[0], Imm_ep[1], Imm_ep[2], Imm_ep[3], Imm_ep[4], Imm_ep[5]),
}
# penguin_means = {
#     'DeepImmuno': (414, 515, 622, 650, 659, 679),
#     'Our Design': (514, 797, 716, 935, 959, 958),
# }

x = np.arange(len(epochs))*0.65  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
color=['#C7A317', '#872657']

fig, ax = plt.subplots(constrained_layout=True, figsize=(7,5))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color[multiplier])
    ax.bar_label(rects, padding=3, fontsize=14)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Training epoch', weight='bold', fontsize=20)
ax.set_ylabel('The number of \nimmunogenic epitopes', weight='bold', fontsize=20)
#ax.set_title('Penguin attributes by species')
ax.set_xticks(x, epochs, fontsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.legend(loc='upper left', fontsize=14)
ax.set_ylim(0, 1100)
plt.savefig(data_file + "results/(New) Imm compare.png", bbox_inches='tight')
plt.show()