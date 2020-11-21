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


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session

# Main #################################################################
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory

for subject in {'Subject1' : dir_path+'/original code/data/Subject1.h5'}: #for now only subject1, later on replace with subjects dictionary from god_config
    for roi in {'VC'}:
        f=open(dir_path+'/results/label-identification/'+subject+'_'+roi+'_'+'label_accuracy'+'.txt','w')
        X=np.load(dir_path+'/data/'+subject+'_'+roi+'_'+'fmri'+'.npy')
        datatype=np.load(dir_path+'/data/'+subject+'_'+'datatype'+'.npy')
        labels=np.load(dir_path+'/data/'+subject+'_'+'labels'+'.npy')
        X=X.reshape(X.shape[0],X.shape[1],X.shape[2],X.shape[3],1)

        input_shape=X.shape

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

        le=LabelEncoder()
        ohe=OneHotEncoder()
        y_train_int=le.fit_transform(y_train.ravel())

        onehot_encoder = OneHotEncoder(sparse=False)
        y_train_int = y_train_int.reshape(len(y_train_int), 1)
        y_train_ohe = ohe.fit_transform(y_train_int).toarray()

        # # invert first example
        # inverted = le.inverse_transform([np.argmax(y_train_ohe[0])])
        # print(inverted)
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy1 = accuracy_score(y_test, y_pred)
        print('Base Accuracy: ' + roi ,accuracy1,file=f)

        def create_model():
            model = Sequential()
            model.add(Conv3D(8,3,activation='relu',input_shape=input_shape[1:]))
            model.add(Conv3D(16,2,activation='relu'))
            model.add(MaxPooling3D(pool_size = (2, 2, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(128, activation = 'relu'))
            model.add(Dropout(0.5))
            model.add(Dense(50, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        model = create_model()

        model.summary()
        model.fit(X_train,y_train_ohe,epochs=3,validation_split=0.1,verbose=0)
        y_pred = model.predict(X_test)

        #Converting predictions to label
        pred = list()
        for i in range(len(y_pred)):
            inverted = le.inverse_transform([np.argmax(y_pred[i])])
            pred.append(inverted[0])

        # print('pred array', pred,file=f)
        print(confusion_matrix(y_true=y_test, y_pred=pred),file=f)
        print("accuracy", accuracy_score(y_test, pred),file=f)
        f.close()
