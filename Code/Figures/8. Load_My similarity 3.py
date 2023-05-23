# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:36:13 2023

@author: ych22001
"""

# sequence matching
import pandas as pd
import numpy as np

#%% Set file directory
data_file = 'D:/PhD thesis/GAN/Github/'

#%%

#data = pd.read_csv('D:/PhD thesis/GCN/My Code/data/gan_a0201.csv')
data = pd.read_csv(data_file + 'Data/neoepitopes/Bladder.4.0_test_mut.csv')
raw = data['peptide'].values
from difflib import SequenceMatcher
def similar(a,b):
    return SequenceMatcher(None,a,b).ratio()

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

import seaborn as sns
#increase font size of all elements
sns.reset_defaults()

"""
Test peptides data
"""
method = 'GAN_RL_imm_select_LG_mean_only'

ep = 'epoch0'
num_epochs = 0

outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '.npy')
df_0=pd.DataFrame({'Max Similarity': whole2})
df_0['Epoch'] = ep

ep = 'epoch20'
num_epochs = 20

outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '.npy')
df_20=pd.DataFrame({'Max Similarity': whole2})
df_20['Epoch'] = ep

ep = 'epoch40'
num_epochs = 40

outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '.npy')
df_40=pd.DataFrame({'Max Similarity': whole2})
df_40['Epoch'] = ep

ep = 'epoch60'
num_epochs = 60

outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '.npy')
df_60=pd.DataFrame({'Max Similarity': whole2})
df_60['Epoch'] = ep

ep = 'epoch80'
num_epochs = 80

outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '.npy')
df_80=pd.DataFrame({'Max Similarity': whole2})
df_80['Epoch'] = ep

ep = 'epoch100'
num_epochs = 100

outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '.npy')
df_100=pd.DataFrame({'Max Similarity': whole2})
df_100['Epoch'] = ep

#%%
df_concate = pd.concat([df_0, df_20, df_40, df_60, df_80, df_100])
# num = df['Max Similarity'].unique()

my_order = np.sort(df_concate['Max Similarity'].unique())
temp = my_order.astype('float64').round(2)
my_order_dec = [format(item, '.2f') for item in my_order]

df_concate_dec = df_concate
df_concate_dec['Max Similarity'] = df_concate['Max Similarity'].apply(lambda x: "{:.2f}".format(x))

g = sns.catplot(data=df_concate_dec, x="Max Similarity", kind="count",
            palette="Set2", edgecolor="black", legend=False,
            order=my_order_dec, height=16, aspect=7,
            hue="Epoch")
g.set(ylim=(0, 600))
# extract the matplotlib axes_subplot objects from the FacetGrid
ax2 = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

# iterate through the axes containers
for c in ax2.containers:
    labels = [f'{v.get_height().astype(int):d}' for v in c]
    ax2.bar_label(c, labels=labels, label_type='edge',fontsize = 30*2, rotation=-30, weight='bold')
#g.set_xticklabels(size = 20)
#g.set_yticklabels(size = 20)
plt.xticks(fontsize=50*2)
plt.yticks(fontsize=50*2)
# plt.title('Epoch '+ str(num_epochs), fontsize=50*3, weight='bold')
plt.ylabel('The number of peptides',fontsize=70*1.5, weight='bold')
plt.xlabel('MaxSimilarity',fontsize=70*1.5, weight='bold')
plt.legend(loc='upper right', ncol = 2,fontsize=70*1.5)
plt.grid()
plt.show()

# sums = df_concate_dec.Smoke.groupby(df.Sex).sum()
# axis('equal');
# pie(sums, labels=sums.index);
# show()

# sum_0 = df_0.groupby(df_0['Max Similarity'])["Max Similarity"].sum()
# labels = df_0['Max Similarity'].unique()
# fig, ax = plt.subplots()
# ax.pie(sum_0, labels=labels)
# plt.show()

# outdir = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/GAN_RL_no_label_similarity_result/' + ep + '/'
# print("outdir is {}".format(outdir))
# plot.savefig(outdir + '/similarity_' + ep + '.pdf')
# np.sum(np.where(np.array(whole2) > 0.75, 1,0))  # 891- 60%