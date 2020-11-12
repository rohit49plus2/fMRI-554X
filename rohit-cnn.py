import numpy as np
import os



dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
####################################################
#ML part
from sklearn.preprocessing import normalize
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
f=open(dir_path+'/test.txt','w')


input_shape=(vc_padded.shape[1],vc_padded.shape[2],vc_padded.shape[3],1)

y=[]
for j in range(len(datatype)):
    if datatype[j]==3:
        y.append(1)
    else:
        y.append(0)

model = DummyClassifier(strategy="most_frequent")
model.fit(vc_padded, y)
y_pred = model.predict(vc_padded)
accuracy1 = accuracy_score(y, y_pred)
print('Base Accuracy',accuracy1,file=f)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session


def create_model():
    model = Sequential()
    model.add(Conv3D(8,3,activation='relu',input_shape=input_shape))
    model.add(Conv3D(4,3,activation='relu'))
    model.add(MaxPooling3D(pool_size = (2, 2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(12, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,verbose=0)
cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=10, random_state=2)
parameters = {'epochs':[10,20,30]
}
clf = GridSearchCV(model, parameters,cv=cv,n_jobs=4)
clf.fit(vc_padded,y)
print('Accuracy: ', clf.best_score_,file=f)
print('Best Parameters: ', clf.best_params_,file=f)
