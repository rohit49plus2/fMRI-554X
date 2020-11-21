import os
import pickle
import pandas as pd
import numpy as np
from god_config import pca
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)
if pca:
    output_file = dir_path+'/results/feature-decoding-pca-final/results.pkl'
else:
    output_file = dir_path+'/results/feature-decoding-final/results.pkl'
with open(output_file, 'rb') as f:
    results = pickle.load(f)
print('Loaded %s' % output_file)

# Show results -----------------------------------------------------
tb_pt = pd.pivot_table(results, index=['roi'], columns=['feature'],
                     values=['catident_correct_rate_percept'], aggfunc=np.mean)
tb_im = pd.pivot_table(results, index=['roi'], columns=['feature'],
                     values=['catident_correct_rate_imagery'], aggfunc=np.mean)

print(tb_pt)
print(tb_im)
