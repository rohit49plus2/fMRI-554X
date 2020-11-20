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

import god_config as config
from sklearn.decomposition import IncrementalPCA
from rohit-image-features import pca
# Main #################################################################
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
def main():
    if pca:
        results_file = dir_path+'/results/feature-decoding-pca-merge/results.pkl'
        output_file = dir_path+'/results/feature-decoding-pca-final/results.pkl'
    else:
        results_file = dir_path+'/results/feature-decoding-merge/results.pkl'
        output_file = dir_path+'/results/feature-decoding-final/results.pkl'

    image_feature_file = config.image_feature_file

    # Load results -----------------------------------------------------
    print('Loading %s' % results_file)
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    data_feature = bdpy.BData(image_feature_file)

    # Category identification ------------------------------------------
    print('Running pair-wise category identification')

    feature_list = results['feature']
    pred_percept = results['predicted_feature_averaged_percept']
    pred_imagery = results['predicted_feature_averaged_imagery']
    cat_label_percept = results['category_label_set_percept']
    cat_label_imagery = results['category_label_set_imagery']
    cat_feature_percept = results['category_feature_averaged_percept']
    cat_feature_imagery = results['category_feature_averaged_imagery']

    ind_cat_other = (data_feature.select('FeatureType') == 4).flatten()

    pwident_cr_pt = []  # Prop correct in pair-wise identification (perception)
    pwident_cr_im = []  # Prop correct in pair-wise identification (imagery)


    for f, fpt, fim, pred_pt, pred_im in zip(feature_list, cat_feature_percept, cat_feature_imagery,
                                             pred_percept, pred_imagery):

        y = data_feature.select(f)
        if pca:
            print('Shape of y before PCA:', y.shape)
            ipca = IncrementalPCA(n_components=20, batch_size=20)
            ipca.fit(y)
            y=ipca.transform(y)
            print('Shape of y after PCA:', y.shape)
            print(y.shape)
        else:
            y = y[:, :100]#take 100 features for time constraint

        feat_other = y[ind_cat_other, :]

        n_unit = fpt.shape[1]
        feat_other = feat_other[:, :n_unit]

        feat_candidate_pt = np.vstack([fpt, feat_other])
        feat_candidate_im = np.vstack([fim, feat_other])

        simmat_pt = corrmat(pred_pt, feat_candidate_pt)
        simmat_im = corrmat(pred_im, feat_candidate_im)

        cr_pt = get_pwident_correctrate(simmat_pt)
        cr_im = get_pwident_correctrate(simmat_im)

        pwident_cr_pt.append(np.mean(cr_pt))
        pwident_cr_im.append(np.mean(cr_im))

    results['catident_correct_rate_percept'] = pwident_cr_pt
    results['catident_correct_rate_imagery'] = pwident_cr_im

    # Save the merged dataframe ----------------------------------------
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print('Saved %s' % output_file)

    # Show results -----------------------------------------------------
    tb_pt = pd.pivot_table(results, index=['roi'], columns=['feature'],
                           values=['catident_correct_rate_percept'], aggfunc=np.mean)
    tb_im = pd.pivot_table(results, index=['roi'], columns=['feature'],
                           values=['catident_correct_rate_imagery'], aggfunc=np.mean)

    from tabulate import tabulate
    print(tb_pt)
    print(tb_im)
    print(tabulate(tb_pt))
    print(tabulate(tb_im))


# Functions ############################################################

def get_pwident_correctrate(simmat):
    '''
    Returns correct rate in pairwise identification

    Parameters
    ----------
    simmat : numpy array [num_prediction * num_category]
        Similarity matrix

    Returns
    -------
    correct_rate : correct rate of pair-wise identification
    '''

    num_pred = simmat.shape[0]
    labels = range(num_pred)

    correct_rate = []
    for i in range(num_pred):
        pred_feat = simmat[i, :]
        correct_feat = pred_feat[labels[i]]
        pred_num = len(pred_feat) - 1
        correct_rate.append((pred_num - np.sum(pred_feat > correct_feat)) / float(pred_num))

    return correct_rate


# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
