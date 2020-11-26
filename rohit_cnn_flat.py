import numpy as np
import os
from god_config import *
import bdpy
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
####################################################
#ML part
from sklearn.preprocessing import normalize
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

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

for subject in {'Subject1'}: #for now only subject1, later on replace with subjects dictionary from god_config
    for roi in rois:
        if not os.path.exists(dir_path+'/results/neural-network/'):
            os.makedirs(dir_path+'/results/neural-network/')
        f=open(dir_path+'/results/neural-network/'+subject+'_'+roi+'_'+'imagination_accuracy-flat'+'.txt','w')
        subject_fmri=bdpy.BData(subjects[subject])
        X=subject_fmri.select(rois[roi])
        del subject_fmri
        datatype=np.load(dir_path+'/data/'+subject+'_'+'datatype'+'.npy')

        input_shape=X.shape

        y=[]
        for j in range(len(datatype)):
            if datatype[j]==3:
                y.append(1)
            else:
                y.append(0)
        y=np.array(y)

        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy1 = accuracy_score(y, y_pred)
        print('Base Accuracy: ' + roi ,accuracy1,file=f)

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=2)

        conf_matrix_list_of_arrays = []
        scores=[]
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            over = SMOTE(random_state=2)
            under = RandomUnderSampler(random_state=2)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            # transform the dataset
            # print("Before SMOLE",X_train.shape)
            X_train=X_train.reshape(X_train.shape[0],-1)
            X_train, y_train = pipeline.fit_resample(X_train, y_train)
            X_train = X_train.reshape((-1,input_shape[1]))
            # print("After SMOLE",X_train.shape)

            ohe=OneHotEncoder()
            y_train=ohe.fit_transform(y_train.reshape(-1,1)).toarray()


            def create_model():
                model = Sequential()
                model.add(Dense(128, activation = 'relu',input_dim=input_shape[1]))
                model.add(Dropout(0.9))
                model.add(Dense(2, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                return model

            model = create_model()

            model.fit(X_train,y_train,epochs=3,verbose=0)
            y_pred = model.predict(X_test)

            #Converting predictions to label
            pred = list()
            for i in range(len(y_pred)):
                pred.append(np.argmax(y_pred[i]))

            conf_matrix = confusion_matrix(y_test, pred)
            conf_matrix_list_of_arrays.append(conf_matrix)
            score=accuracy_score(y_test, pred)
            scores.append(score)

        mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
        print(mean_of_conf_matrix_arrays,file=f)
        print('Accuracy: %.7f (%.7f)' % (np.mean(scores), np.std(scores)),file=f)

        f.close()
