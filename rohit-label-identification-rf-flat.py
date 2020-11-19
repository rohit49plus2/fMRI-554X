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


from sklearn.preprocessing import normalize
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPooling3D
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.backend import clear_session

# Main #################################################################
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory

for subject in {'Subject1' : dir_path+'/original code/data/Subject1.h5'}: #for now only subject1, later on replace with subjects dictionary from god_config
    for roi in {'VC'}:
        f=open(dir_path+'/results/label-identification/'+subject+'_'+roi+'_'+'label_accuracy-RF-flat'+'.txt','w')
        subject1=bdpy.BData(subjects[subject])
        X=subject1.select(rois[roi])
        del subject1
        datatype=np.load(dir_path+'/data/'+subject+'_'+'datatype'+'.npy')
        labels=np.load(dir_path+'/data/'+subject+'_'+'labels'+'.npy')
        # X=X.reshape(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])

        # input_shape=X.shape

        i_train = (datatype == 1).flatten()    # Index for training 1200 trials
        i_test_pt = (datatype == 2).flatten()  # Index for perception test 35 runs of 50 images = 1750
        i_test_im = (datatype == 3).flatten()  # Index for imagery test 20 runs of 25 images
        i_test=i_test_im + i_test_pt

        all_labels = np.vstack([int(n) for n in labels.flatten()])
        all_uniq=np.unique(all_labels)

        y=all_labels

        X_train = X[i_test_pt]
        X_test = X[i_test_im]
        y_train = y[i_test_pt]
        y_test = y[i_test_im]
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        model = RandomForestClassifier()
        # evaluate model
        # cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=10, random_state=2)
        parameters = {'max_depth':[1,2,3,4,5],
            'n_estimators': [10,50,100],
            'max_features': [1,2,3,4,5]
        }
        clf = GridSearchCV(model, parameters,cv=5,n_jobs=4)
        clf.fit(X_train,y_train)
        print('CV accuracy: ', clf.best_score_,file=f)
        print('Best Parameters: ', clf.best_params_,file=f)
        # print('\n\ncv results: ', clf.cv_results_)


        model = RandomForestClassifier(max_depth=clf.best_params_['max_depth'],n_estimators=clf.best_params_['n_estimators'],max_features=clf.best_params_['max_features'])

        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        print(confusion_matrix(y_true=y_test, y_pred=pred),file=f)
        print("accuracy", accuracy_score(y_test, pred),file=f)
        print("precision", precision_score(y_test, pred,average='micro'),file=f)

        f.close()
