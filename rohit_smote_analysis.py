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
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


for subject in {'Subject1'}: #for now only subject1, later on replace with subjects dictionary from god_config
    for roi in ['VC']:
        # f=open(dir_path+'/results/without-convolution/'+subject+'_'+roi+'_'+'imagination_accuracy-flat'+'.txt','w')
        subject_fmri=bdpy.BData(subjects[subject])
        X=subject_fmri.select(rois[roi])
        voxel_x = subject_fmri.get_metadata('voxel_x', where=area[roi])
        voxel_y = subject_fmri.get_metadata('voxel_y', where=area[roi])
        voxel_z = subject_fmri.get_metadata('voxel_z', where=area[roi])
        del subject_fmri
        datatype=np.load(dir_path+'/data/'+subject+'_'+'datatype'+'.npy')
        # X=X.reshape(X.shape[0],X.shape[1],1)

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
        print('Base Accuracy: ' + roi ,accuracy1)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

        true_im=X_train[y_train==1]
        true_pt=X_train[y_train==0]

        over = SMOTE(random_state=2)
        under = RandomUnderSampler(random_state=2)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        # transform the dataset
        print("Before SMOLE",X_train.shape)
        X_train=X_train.reshape(X_train.shape[0],-1)
        X_train, y_train = pipeline.fit_resample(X_train, y_train)
        X_train = X_train.reshape((-1,input_shape[1]))
        print("After SMOLE",X_train.shape)

        smote_im=X_train[y_train==1]
        smote_pt=X_train[y_train==0]

        smote_av = np.mean(X_train,axis=0)
        smote_im_av = np.mean(smote_im,axis=0)
        smote_pt_av = np.mean(smote_pt,axis=0)

        true_av = np.mean(X_train,axis=0)
        true_im_av = np.mean(true_im,axis=0)
        true_pt_av = np.mean(true_pt,axis=0)


        fig = plt.figure(figsize =(2, 3))

        ax = fig.add_subplot(2, 3, 1,projection="3d")
        plot=ax.scatter(voxel_x,voxel_y,voxel_z,s=20,c=true_av,cmap='coolwarm',vmin=-10,vmax=+10)
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        ax.set_title("Average of true data in " + roi+ " of " + subject)
        fig.colorbar(plot,shrink=1)

        ax = fig.add_subplot(2, 3, 2,projection="3d")
        plot=ax.scatter(voxel_x,voxel_y,voxel_z,s=20,c=true_pt_av,cmap='coolwarm',vmin=-0.8,vmax=+0.8)
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        ax.set_title("Average of true perception data in " + roi+ " of " + subject)
        fig.colorbar(plot,shrink=1)

        ax = fig.add_subplot(2, 3, 3,projection="3d")
        plot=ax.scatter(voxel_x,voxel_y,voxel_z,s=20,c=true_im_av,cmap='coolwarm',vmin=-10,vmax=+10)
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        ax.set_title("Average of true imagined data in " + roi+ " of " + subject)
        fig.colorbar(plot,shrink=1)

        ax = fig.add_subplot(2, 3, 4, projection='3d')
        plot=ax.scatter(voxel_x,voxel_y,voxel_z,s=20,c=smote_av,cmap='coolwarm',vmin=-10,vmax=+10)
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        ax.set_title("Average of synthetic data in " + roi+ " of " + subject)
        fig.colorbar(plot,shrink=1)


        ax = fig.add_subplot(2, 3, 5, projection='3d')
        plot=ax.scatter(voxel_x,voxel_y,voxel_z,s=20,c=smote_pt_av,cmap='coolwarm',vmin=-0.8,vmax=+0.8)
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        ax.set_title("Average of synthetic perception data in " + roi+ " of " + subject)
        fig.colorbar(plot,shrink=1)


        ax = fig.add_subplot(2, 3, 6, projection='3d')
        plot=ax.scatter(voxel_x,voxel_y,voxel_z,s=20,c=smote_im_av,cmap='coolwarm',vmin=-10,vmax=+10)
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        ax.set_title("Average of synthetic imagined data in " + roi+ " of " + subject)
        fig.colorbar(plot,shrink=1)

        plt.show()
