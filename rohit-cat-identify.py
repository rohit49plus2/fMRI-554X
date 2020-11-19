'''
Object category identification

This file is a part of GenericDecoding_demo.
'''


from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd

import bdpy
from bdpy.stats import corrmat

from god_config import *

from sklearn.decomposition import IncrementalPCA

# Main #################################################################
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
results_dir = dir_path+'/results/feature-decoding/'
output_file = dir_path+'/results/feature-decoding-merged/results.pkl'

# Load results -----------------------------------------------------
result_list = []
image_features=bdpy.BData(image_feature_file)

for subject in {'Subject1'}:
    for feat in features:
        y = image_features.select(feat)
        print('Shape of y before PCA:', y.shape)
        ipca = IncrementalPCA(n_components=5, batch_size=120)
        ipca.fit(y)
        y=ipca.transform(y)
        print('Shape of y after PCA:', y.shape)
        print(y.shape)


        ind_cat_other = (image_features.select('FeatureType') == 4).flatten()
        other_category_features=y[ind_cat_other, :]
        for roi in rois:
            rf_full = dir_path+'/results/feature-decoding/'+subject+'_'+roi+'_'+feat+'_'+'decode_results.pkl'
            print('Loading %s' % rf_full)
            with open(rf_full, 'rb') as f:
                res = pickle.load(f)
            test_label=res['test_label']
            true_feature_pt=res['true_feature'][0]
            true_feature_im=res['true_feature'][1]
            predicted_feature=res['predicted_feature']
            av_feature_pt=dict(zip(res['category_label_set'][0],res['category_feature_averaged'][0]))
            av_feature_im=dict(zip(res['category_label_set'][1],res['category_feature_averaged'][1]))

            #identification accuracy test - check correlation between predicted feature and category averaged feature vs random feature and category averaged feature
            num_correct_identify_pt=[]
            num_correct_identify_im=[]



    result_list.append(res)
