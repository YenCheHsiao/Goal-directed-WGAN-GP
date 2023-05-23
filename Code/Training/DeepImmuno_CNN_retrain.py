# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:32:57 2023

@author: xiaoyenche
"""
'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)
'''

from DeepImmuno_CNN_define import *
# all the functions can be inspected in cnn_train.py file (in reproduce folder as well), this is just a notebook showing how to train the model

# set your working directory to the reproduce folder
os.chdir('D:/PhD thesis/GAN/Github/Data/DeepImmuno')

# load after_pca, this stores the features to encode amino acid/peptides
after_pca = np.loadtxt('./after_pca.txt')

# load training dataset
ori = pd.read_csv('./remove0123_sample100.csv')
ori = ori.sample(frac=1, replace=False).set_index(pd.Index(np.arange(ori.shape[0])))
ori.head()

# load hla paratope, this is used for encoding HLA sequence
hla = pd.read_csv('./hla2paratopeTable_aligned.txt', sep='\t')

# some pre-processing to parse the inputs to be compatible with deep learning foramt
hla_dic = hla_df_to_dic(hla)
inventory = list(hla_dic.keys())
dic_inventory = dict_inventory(inventory)
dataset = construct_aaindex(ori, hla_dic, after_pca,dic_inventory)
input1 = pull_peptide_aaindex(dataset)
input2 = pull_hla_aaindex(dataset)
label = pull_label_aaindex_continuous(dataset)

# representing peptide
input1.shape # (8971, 10, 12, 1)

# representing hla
input2.shape # (8971, 46, 12, 1)

# representing label
label.shape # (8971, 1)

# start training, first split 90% train and 10% internal validation
array = np.arange(len(dataset))
train_index = np.random.choice(array,int(len(dataset)*0.9),replace=False)
valid_index = [item for item in array if item not in train_index]
input1_train = input1[train_index]
input1_valid = input1[valid_index]
input2_train = input2[train_index]
input2_valid = input2[valid_index]
label_train = label[train_index]
label_valid = label[valid_index]

# main training steps
cnn_model = seperateCNN()
# cnn_model.compile(
#     loss=keras.losses.MeanSquaredError(),
#     optimizer=keras.optimizers.Adam(lr=0.0001),
#     metrics=['accuracy'])
# https://stackoverflow.com/questions/45632549/why-is-the-accuracy-for-my-keras-model-always-0-when-training
cnn_model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(lr=0.0001),
    metrics=['mean_squared_error'])

callback_val = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,restore_best_weights=False)
callback_train = keras.callbacks.EarlyStopping(monitor='loss',patience=2,restore_best_weights=False)

checkpoint_path = "D:/PhD thesis/GAN/Github/Data/DeepImmuno/training_1/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# https://www.tensorflow.org/tutorials/keras/save_and_load
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
history = cnn_model.fit(
    x=[input1_train,input2_train],   # feed a list into
    y=label_train,
    validation_data = ([input1_valid,input2_valid],label_valid),
    batch_size=128,
    epochs=200,
    class_weight = {0:0.5,1:0.5},   # I have 20% positive and 80% negative in my training data
    callbacks = [callback_val,callback_train,cp_callback])

draw_history(history)