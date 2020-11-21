# Test code to see how the data files are stored
# analysis_FeaturePrediction.py in the /original code folder is a good way to understand
from __future__ import print_function
import os
import sys
import pickle
from itertools import product
from time import time
import numpy as np
import pandas as pd
from scipy import stats
from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression  # For quick demo
import bdpy
from bdpy.bdata import concat_dataset
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.dataform import append_dataframe
from bdpy.distcomp import DistComp
from god_config import * #import metadata from this script

from tqdm import tqdm


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


from sklearn.preprocessing import OneHotEncoder, normalize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session

def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


#feature prediction Function
def feature_prediction(subject, roi, y_train, y_test, n_voxel=500, n_iter=200):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''

    n_unit = y_train.shape[1]

    subject1=bdpy.BData(subjects[subject])
    X=subject1.select(rois[roi])
    del subject1

    input_shape=X.shape

    # Feature prediction for each unit
    print('Running feature prediction')

    y_true_list = []
    y_pred_list = []
    for i in range(n_unit):

        print('Unit %03d' % (i + 1))
        start_time = time()

        #get unit fmri
        X_train = X[i_train]
        X_test = X[i_test]

        # Get unit image features
        y_train_unit = y_train[:, i]
        y_test_unit =  y_test[:, i]

        # Normalize image features for training (y_train_unit)
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # over = SMOTE(random_state=2)
        # under = RandomUnderSampler(random_state=2)
        # steps = [('o', over), ('u', under)]
        # pipeline = Pipeline(steps=steps)
        # # transform the dataset
        # print("Before SMOLE",X_train.shape)
        # X_train, y_train_unit = pipeline.fit_resample(X_train, y_train_unit)
        # X_train = X_train.reshape((-1,input_shape[1],input_shape[2],input_shape[3],input_shape[4]))
        # print("After SMOLE",X_train.shape)

        def create_model():
            model = Sequential()
            model.add(Dense(128, activation = 'relu',input_dim=input_shape[1]))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        model = create_model()

        model.fit(X_train,y_train_unit,epochs=2,verbose=0)
        y_pred = model.predict(X_test)
        # Denormalize predicted features
        y_pred = y_pred * norm_scale_y + norm_mean_y
        y_pred=y_pred.ravel()
        y_true_list.append(y_test_unit)
        y_pred_list.append(y_pred)

        print('Time: %.3f sec' % (time() - start_time))

    # Create numpy arrays for return values
    # print(y_pred_list)
    # print(np.vstack(y_pred_list).shape)
    y_predicted = np.vstack(y_pred_list).T
    y_true = np.vstack(y_true_list).T

    return y_predicted, y_true


# Main #################################################################
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory


for subject in {'Subject1' : dir_path+'/original code/data/Subject1.h5'}: #for now only subject1, later on replace with subjects dictionary from god_config
    #Load image features
    image_features=bdpy.BData(image_feature_file)
    datatype=np.load(dir_path+'/data/'+subject+'_'+'datatype'+'.npy')
    labels=np.load(dir_path+'/data/'+subject+'_'+'labels'+'.npy')
    # print("voxel_data shape: ", voxel_data.shape)
    i_train = (datatype == 1).flatten()    # Index for training 1200 trials
    i_test_pt = (datatype == 2).flatten()  # Index for perception test 35 runs of 50 images = 1750
    i_test_im = (datatype == 3).flatten()  # Index for imagery test 20 runs of 25 images
    i_test=i_test_im + i_test_pt
    for feat in  features:
        # f=open(dir_path+'/results/feature-decoding/texts/'+subject+'_'+roi+'_'+feat+'_'+'feature-decoding'+'.txt','w')
        y = image_features.select(feat)             # Image features
        if pca:
            from sklearn.decomposition import IncrementalPCA
            print('Shape of y before PCA:', y.shape)
            ipca = IncrementalPCA(n_components=20, batch_size=20)
            ipca.fit(y)
            y=ipca.transform(y)
            print('Shape of y after PCA:', y.shape)
        else:
            y = y[:, :100]#take 100 features for time constraint

        y_label = image_features.select('ImageID')  # Image labels

        y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

        y_train = y_sorted[i_train, :]
        y_test = y_sorted[i_test, :]

        for roi in ['VC']:

            # Feature prediction
            pred_y, true_y = feature_prediction(subject, roi, y_train, y_test)


            i_pt = i_test_pt[i_test]  # Index for perception test within test
            i_im = i_test_im[i_test]  # Index for imagery test within test

            print(pred_y.shape)
            print(i_pt.shape)

            pred_y_pt = pred_y[i_pt, :]
            pred_y_im = pred_y[i_im, :]

            true_y_pt = true_y[i_pt, :]
            true_y_im = true_y[i_im, :]

            test_label_pt = labels[i_test_pt, :].flatten()
            test_label_im = labels[i_test_im, :].flatten()

            pred_y_pt_av, true_y_pt_av, test_label_set_pt = get_averaged_feature(pred_y_pt, true_y_pt, test_label_pt)
            pred_y_im_av, true_y_im_av, test_label_set_im = get_averaged_feature(pred_y_im, true_y_im, test_label_im)

            # Get category averaged features
            catlabels_pt = np.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
            catlabels_im = np.vstack([int(n) for n in test_label_im])  # Category labels (imagery test)
            catlabels_set_pt = np.unique(catlabels_pt)                 # Category label set (perception test)
            catlabels_set_im = np.unique(catlabels_im)                 # Category label set (imagery test)
            y_catlabels = image_features.select('CatID')   # Category labels in image features
            ind_catave = (image_features.select('FeatureType') == 3).flatten()


            y_catave_pt = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_pt)
            y_catave_im = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_im)

            # Prepare result dataframe
            results = pd.DataFrame({'subject' : [subject, subject],
                                    'roi' : [roi, roi],
                                    'feature' : [feat, feat],
                                    'test_type' : ['perception', 'imagery'],
                                    'true_feature': [true_y_pt, true_y_im],
                                    'predicted_feature': [pred_y_pt, pred_y_im],
                                    'test_label' : [test_label_pt, test_label_im],
                                    'test_label_set' : [test_label_set_pt, test_label_set_im],
                                    'true_feature_averaged' : [true_y_pt_av, true_y_im_av],
                                    'predicted_feature_averaged' : [pred_y_pt_av, pred_y_im_av],
                                    'category_label_set' : [catlabels_set_pt, catlabels_set_im],
                                    'category_feature_averaged' : [y_catave_pt, y_catave_im]})

            # print('catlabels_set_pt size',catlabels_set_pt.shape)
            # print('catlabels_set_im size',catlabels_set_im.shape)
            # print('true_y_pt_av size',true_y_pt_av.shape)
            # print('true_y_im_av size',true_y_im_av.shape)
            # print('pred_y_pt_av size',pred_y_pt_av.shape)
            # print('pred_y_im_av size',pred_y_im_av.shape)
            # print('y_catave_pt size',y_catave_pt.shape)
            # print('y_catave_im size',y_catave_im.shape)
            if pca:
                res=dir_path+'/results/feature-decoding-pca/'+subject+'_'+roi+'_'+feat+'_'+'decode_results.pkl'
            else:
                res=dir_path+'/results/feature-decoding/'+subject+'_'+roi+'_'+feat+'_'+'decode_results.pkl'
            makedir_ifnot(os.path.dirname(res))

            with open(res, 'wb') as f:
                pickle.dump(results, f)

            print('Saved %s' % res)
