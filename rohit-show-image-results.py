import os
import pickle
import pandas as pd
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
output_file = dir_path+'/results/feature-decoding-final/results.pkl'
with open(output_file, 'rb') as f:
    results = pickle.load(f)
print('Loaded %s' % output_file)

# Show results -----------------------------------------------------
tb_pt = pd.pivot_table(results, index=['roi'], columns=['feature'],
                     values=['catident_correct_rate_percept'], aggfunc=np.mean)
tb_im = pd.pivot_table(results, index=['roi'], columns=['feature'],
                     values=['catident_correct_rate_imagery'], aggfunc=np.mean)

from tabulate import tabulate
print(tabulate(tb_pt))
print(tabulate(tb_im))
