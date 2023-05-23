# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:18:01 2023

@author: ych22001
"""

# sequence matching
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Set file directory
data_file = 'D:/PhD thesis/GAN/Github/'

#%%

"""
No predictor peptides data (1. GAN+CNN (no label).py)
"""
ep = 'epoch1000'
batch_size = 10000 # 60000 peptides
method = 'GAN_RL_old'
num_epochs = 399
num_epochs = 1000

outdir_1 = data_file + 'Data/In paper/' + method + '/' + ep + '/'
outname_1 = 'deepimmuno-GANRL-bladder_cancer-scored_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt'

# data_path = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/' + method + '/' + ep + '/' + 'deepimmuno-GANRL-bladder_cancer-scored_' + ep +'.txt'
# data_path = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/' + method + '/' + ep + '/' + 'deepimmuno-GANRL-bladder_cancer-scored_' + ep + '-batch' + str(batch_size) + '.txt'
data_path = outdir_1 + outname_1
generate = pd.read_csv(data_path,sep='\t')
imm = generate['immunogenicity'].values

#generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
#generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
seq = generate['peptide'].values

# outdir = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/GAN_RL_no_label_similarity_result/' + ep + '/'
# outdir = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/' + method + '/' + ep + '/'
outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
# whole2 = np.load(outdir + '/MaxSimilarity_' + ep + '.npy')
# whole2 = np.load(outdir + '/MaxSimilarity_' + ep + '-batch' + str(batch_size) + '.npy')
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.npy')

df = pd.DataFrame({'immunogenicity':imm, 'MaxSimilarity':whole2})
df.plot('immunogenicity', 'MaxSimilarity', kind='scatter')
plt.show()

"""
With predictor peptides data (1-10. (12-select only) GAN+CNN (score select LG mean))
"""
ep = 'epoch1000'
batch_size = 10000 # 60000 peptides
method = 'GAN_RL_imm_select_LG_mean_only'

outdir_1 = data_file + 'Data/In paper/' + method + '/' + ep + '/'
outname_1 = 'deepimmuno-GANRL-bladder_cancer-scored_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt'

# data_path = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/' + method + '/' + ep + '/' + 'deepimmuno-GANRL-bladder_cancer-scored_' + ep +'.txt'
# data_path = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/' + method + '/' + ep + '/' + 'deepimmuno-GANRL-bladder_cancer-scored_' + ep + '-batch' + str(batch_size) + '.txt'
data_path = outdir_1 + outname_1
generate = pd.read_csv(data_path,sep='\t')
imm = generate['immunogenicity'].values

#generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
#generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
seq = generate['peptide'].values

# outdir = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/GAN_RL_no_label_similarity_result/' + ep + '/'
# outdir = 'C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/Test/' + method + '/' + ep + '/'
outdir = data_file + 'Data/In paper/' + method + '/' + ep + '/'
# whole2 = np.load(outdir + '/MaxSimilarity_' + ep + '.npy')
whole2 = np.load(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.npy')

df2 = pd.DataFrame({'immunogenicity':imm, 'MaxSimilarity':whole2})
df2.plot('immunogenicity', 'MaxSimilarity', kind='scatter')
plt.show()

#%% Scatter plot
min_imm = pd.concat([df, df2])['immunogenicity'].min()
max_imm = pd.concat([df, df2])['immunogenicity'].max()

min_sim = pd.concat([df, df2])['MaxSimilarity'].min()
max_sim = pd.concat([df, df2])['MaxSimilarity'].max()

plt.scatter(df2['immunogenicity'], df2['MaxSimilarity'],s=50,marker='^',alpha=0.5,
            edgecolor ="purple")
plt.scatter(df['immunogenicity'], df['MaxSimilarity'],s=50,marker='s',alpha=0.5,
            edgecolor ="black")
plt.legend(['Our method', 'No predictor'], loc='best', fontsize=10)
plt.xlabel('Immunogenicity', fontsize=10)
plt.ylabel('MaxSimilarity', fontsize=10)
plt.show()

x_offset = 0.05
y_offset = 0.05

plt.figure(figsize=(9.0, 5.8))
plt.scatter(df['immunogenicity'], df['MaxSimilarity'], c='#1f77b4',s=50,marker='s',alpha=0.05
            , edgecolor ="black"
            )
plt.xlim(min_imm - x_offset, max_imm + x_offset) 
plt.ylim(min_sim - y_offset, max_sim + y_offset) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('WGAN-GP without\nimmunogenicity predictor', fontsize=30, weight='bold')
plt.xlabel('Immunogenicity score', fontsize=30, weight='bold')
plt.ylabel('MaxSimilarity', fontsize=30, weight='bold')
plt.grid()
plt.show()

plt.figure(figsize=(9.0, 5.8))
plt.scatter(df2['immunogenicity'], df2['MaxSimilarity'], c='#ff7f0e' ,s=50,marker='s',alpha=0.05
            , edgecolor ="black"
            )
plt.xlim(min_imm - x_offset, max_imm + x_offset) 
plt.ylim(min_sim - y_offset, max_sim + y_offset) 
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Goal-directed WGAN-GP with\nimmunogenicity predictor', fontsize=30, weight='bold')
plt.xlabel('Immunogenicity score', fontsize=30, weight='bold')
plt.ylabel('MaxSimilarity', fontsize=30, weight='bold')
plt.grid()
plt.show()

#%% Violin plot
df_labled = df
df_labled['Method'] = 'WGAN-GP'
df2_labled = df2
df2_labled['Method'] = 'Goal-directed WGAN-GP'
df_concate = pd.concat([df2_labled, df_labled])
df_concate['Immunogenicity score'] = df_concate['immunogenicity']

my_order = -np.sort(-df_concate['MaxSimilarity'].unique())
# set 2 decimal place value in seaborn
# https://stackoverflow.com/questions/57441873/seaborn-countplot-rotation-and-formatting-to-decimal-places
fig, ax1 = plt.subplots(
    figsize=(7.4,4.8)
                        )
# print(sns.color_palette("pastel").as_hex())
# #a1c9f4
# #ffb482
sns.violinplot(data=df_concate, x="Immunogenicity score", y="MaxSimilarity"
               , hue="Method", split=True#, palette="pastel"
               , orient='h', order=my_order
               , ax=ax1, palette=['#ffb482', '#a1c9f4'])
ax1.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax1.get_yticklabels()])
plt.legend(bbox_to_anchor=(0., 1.15), loc='upper left', ncol=2,fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Immunogenicity score', fontsize=20, weight='bold')
plt.ylabel('MaxSimilarity', fontsize=20, weight='bold')
plt.grid()
plt.show()

#%% Histogram
# https://seaborn.pydata.org/archive/0.11/tutorial/categorical.html
temp = my_order.astype('float64').round(2)

my_order_dec = [format(item, '.2f') for item in my_order]

df_concate_dec = df_concate
df_concate_dec['MaxSimilarity'] = df_concate['MaxSimilarity'].apply(lambda x: "{:.2f}".format(x))

g = sns.catplot(data=df_concate_dec, y="MaxSimilarity", hue="Method", kind="count",
            palette=['#ffb482', '#a1c9f4'], edgecolor=".6", legend=False, order=my_order_dec, height=8)

# extract the matplotlib axes_subplot objects from the FacetGrid
ax2 = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

# iterate through the axes containers
for c in ax2.containers:
    labels = [f'{v.get_width().astype(int):d}' for v in c]
    ax2.bar_label(c, labels=labels, label_type='edge',fontsize = 20)
plt.legend(['Goal-directed WGAN-GP','WGAN-GP'], loc='upper right',fontsize = 25)
g.set_xticklabels(size = 30)
g.set_yticklabels(size = 30)
plt.ylabel('MaxSimilarity',fontsize=30, weight='bold')
plt.xlabel('The number of peptides',fontsize=30, weight='bold')
plt.grid()
# ax2.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax2.get_yticklabels()])
plt.show()

#%% Box plot
plt.rcParams['figure.figsize']
# [6.4, 4.8]
# print(sns.color_palette().as_hex())
# blue: #1f77b4
# orange: #ff7f0e
fig3, ax3 = plt.subplots(figsize=(6.6, 4.8))
sns.boxplot(data=df_concate, x="Method", y="Immunogenicity score",saturation=0.5,width=0.5,palette=['#ff7f0e', '#1f77b4'], ax=ax3)
sns.stripplot(data=df_concate, x="Method", y="Immunogenicity score",palette=['#ffb482', '#a1c9f4'], size=1, alpha=0.5, ax=ax3)
# iterate through the axes containers
for c in ax3.xaxis.get_major_ticks():
    c.label.set_fontsize(15)
for c in ax3.yaxis.get_major_ticks():
    c.label.set_fontsize(15)
plt.ylabel('Immunogenicity score',fontsize=20, weight='bold')
plt.xlabel('Methods',fontsize=20, weight='bold')
plt.grid()
plt.show()