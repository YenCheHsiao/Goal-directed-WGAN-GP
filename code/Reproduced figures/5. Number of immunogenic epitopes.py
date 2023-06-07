# -*- coding: utf-8 -*-
'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)
'''
"""
Created on Tue Feb 28 18:04:37 2023

@author: xiaoyenche
"""

# source code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

import matplotlib.pyplot as plt
import numpy as np

epochs = ("0", "20", "40", "60", "80", "100")
penguin_means = {
    'DeepImmuno$^{19}$': (414, 515, 622, 650, 659, 679),
    'Our Design': (514, 797, 716, 935, 959, 958),
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

plt.show()