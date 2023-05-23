# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:11:20 2023

@author: ych22001
"""

import pandas as pd
# Database: http://biopharm.zju.edu.cn/tsnadb/
data_file = 'D:/PhD thesis/GAN/Github/'
Bladder = pd.read_csv(data_file + 'Data/neoepitopes/Bladder.4.0.txt',sep='\t')
Bladder.columns
Bladder['HLA']
Bladder['wild_peptide']
temp1 = Bladder[Bladder['HLA'].isin(['A*02:01'])]['wild_peptide']
temp2 = []
for i in range(len(temp1)):
    if len(temp1.iloc[i])<11:
        if len(temp1.iloc[i])>8:
            temp2.append(temp1.iloc[i])
            
import os
# C:\Users\ych22001\OneDrive - University of Connecticut\Documents\2. GCN\Vaccine\Code\Test\Database\Bladder.4.0
output_dir = data_file + 'Data/neoepitopes/'
df = pd.DataFrame({'peptide': temp2, 'HLA': ['HLA-A*0201' for i in range(len(temp2))]})
df.to_csv(os.path.join(output_dir, 'Bladder.4.0_test_mut.csv'), index=None)