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


from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session

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

    X=np.load(dir_path+'/data/'+subject+'_'+roi+'_'+'fmri'+'.npy')
    X=X.reshape(X.shape[0],X.shape[1],X.shape[2],X.shape[3],1)

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
            model.add(Conv3D(8,3,activation='relu',input_shape=input_shape[1:]))
            model.add(Conv3D(16,2,activation='relu'))
            model.add(MaxPooling3D(pool_size = (2, 2, 2)))
            model.add(Dropout(0.6))
            model.add(Flatten())
            model.add(Dense(128, activation = 'relu'))
            model.add(Dropout(0.9))
            model.add(Dense(1, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        model = create_model()

        model.fit(X_train,y_train_unit,epochs=2,verbose=0)
        y_pred = model.predict(X_test)
        # Denormalize predicted features
        y_pred = y_pred * norm_scale_y + norm_mean_y

        y_true_list.append(y_test_unit)
        y_pred_list.append(y_pred)

        print('Time: %.3f sec' % (time() - start_time))

    # Create numpy arrays for return values
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
    for roi in {'VC'}:
        for feat in ['cnn8']:
            y = image_features.select(feat)             # Image features
            y_label = image_features.select('ImageID')  # Image labels

            y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

            y_train = y_sorted[i_train, :]
            y_test = y_sorted[i_test, :]

            # Feature prediction
            pred_y, true_y = feature_prediction(subject, roi, y_train, y_test)

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

            del image_features

            y_catave_pt = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_pt)
            y_catave_im = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_im)

            # Prepare result dataframe
            results = pd.DataFrame({'subject' : [sbj, sbj],
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

            res=dir_path+'/data/'+subject+'_'+roi+'_'+'image_results.pkl'
            makedir_ifnot(os.path.dirname(res))
            with open(res, 'wb') as f:
                pickle.dump(results, f)

            print('Saved %s' % results_file)
