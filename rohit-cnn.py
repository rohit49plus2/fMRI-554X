import numpy as np
import os


dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
####################################################
#ML part
from sklearn.preprocessing import normalize
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
f=open(dir_path+'/rohit-test.txt','w')

X=np.load(dir_path+'/data/'+'Subject1'+'_'+'VC'+'_'+'fmri'+'.npy')
datatype=np.load(dir_path+'/data/'+'Subject1'+'_'+'datatype'+'.npy')
X=X.reshape(X.shape[0],X.shape[1],X.shape[2],X.shape[3],1)

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
print('Base Accuracy',accuracy1,file=f)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
over = SMOTE()
under = RandomUnderSampler()
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
print("Before SMOLE",X_train.shape)
X_train=X_train.reshape(X_train.shape[0],-1)
X_train, y_train = pipeline.fit_resample(X_train, y_train)
X_train = X_train.reshape((-1,input_shape[1],input_shape[2],input_shape[3],input_shape[4]))
print("After SMOLE",X_train.shape)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
y_train=ohe.fit_transform(y_train.reshape(-1,1)).toarray()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.backend import clear_session


def create_model():
    model = Sequential()
    model.add(Conv3D(8,3,activation='relu',input_shape=input_shape[1:]))
    model.add(Conv3D(16,2,activation='relu'))
    model.add(MaxPooling3D(pool_size = (2, 2, 2)))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.9))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()

model.fit(X_train,y_train,epochs=1)
y_pred = model.predict(X_test)

#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

print(confusion_matrix(y_true=y_test, y_pred=pred),file=f)
print("accuracy", accuracy_score(y_test, pred),file=f)
print("precision", precision_score(y_test, pred,average='micro'),file=f)
f.close()

# cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=10, random_state=2)
# parameters = {'epochs':[10,20,30]
# }
# clf = GridSearchCV(model, parameters,cv=cv,n_jobs=4)
# clf.fit(X,y)
# print('Accuracy: ', clf.best_score_,file=f)
# print('Best Parameters: ', clf.best_params_,file=f)
