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
# Main #################################################################
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory

for subject in {'Subject1' : dir_path+'/original code/data/Subject1.h5'}: #for now only subject1, later on replace with subjects dictionary from god_config
    #Load subject's data
    subject1=bdpy.BData(subjects[subject])
    #Load image features
    # image_features=bdpy.BData(image_feature_file)
    # subject1.show_metadata()
    datatype = subject1.select('DataType')   # Data type
    labels = subject1.select('stimulus_id')  # Image labels in brain data
    # voxel_data = subject1.select('VoxelData')  # Image labels in brain data
    # voxel_data = subject1.select('VolInds')  # Image labels in brain data
    # print("datatype shape: ", datatype.shape)
    # print("labels shape: ", labels.shape)
    # print("voxel_data shape: ", voxel_data.shape)
    # i_train = (datatype == 1).flatten()    # Index for training 1200 trials
    # i_test_pt = (datatype == 2).flatten()  # Index for perception test 35 runs of 50 images = 1750
    # i_test_im = (datatype == 3).flatten()  # Index for imagery test 20 runs of 25 images
    # print("indexes for imagery:", i_test_im)
    # print("labels unique",np.unique(labels).shape)
    # print("i_train sum",i_train.sum())
    # print("i_test_pt sum",i_test_pt.sum())
    # print("i_test_im sum",i_test_im.sum())
    np.save(dir_path+'/data/'+subject+'_'+'datatype',datatype)
    np.save(dir_path+'/data/'+subject+'_'+'labels',labels)


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    def plot_fmri_colors(roi,num):
        """
        can pass roi as string like 'VC', takes area dictionary from god config
        plots fmri values in that region for trial number num
        """
        voxel_x = subject1.get_metadata('voxel_x', where=area[roi])
        voxel_y = subject1.get_metadata('voxel_y', where=area[roi])
        voxel_z = subject1.get_metadata('voxel_z', where=area[roi])
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        plot=ax.scatter(voxel_x,voxel_y,voxel_z,c=subject1.select(rois[roi])[num][:],cmap='coolwarm')
        ax.set_xlabel('X Axes')
        ax.set_ylabel('Y Axes')
        ax.set_zlabel('Z Axes')
        ax.set_title(roi)

        fig.colorbar(plot)
        plt.show()

    def coordinates(roi):
        """
        can pass roi as string like 'VC', takes area dictionary from god config
        returns array of size (num_voxels in that roi,) with coordinates.
        """
        voxel_x = subject1.get_metadata('voxel_x', where=area[roi])
        voxel_y = subject1.get_metadata('voxel_y', where=area[roi])
        voxel_z = subject1.get_metadata('voxel_z', where=area[roi])
        co=[]
        for i in range(len(voxel_x)):
            co.append((voxel_x[i],voxel_y[i],voxel_z[i]))
        return(co)


    full_voxel_x = subject1.get_metadata('voxel_x', where=area['VC'])
    full_voxel_y = subject1.get_metadata('voxel_y', where=area['VC'])
    full_voxel_z = subject1.get_metadata('voxel_z', where=area['VC'])
    full_x_shape=int((full_voxel_x.max()-full_voxel_x.min())/3+1)
    full_y_shape=int((full_voxel_y.max()-full_voxel_y.min())/3+1)
    full_z_shape=int((full_voxel_z.max()-full_voxel_z.min())/3+1)
    def padded_fmri(roi):
        """
        pass roi as string like 'VC', takes area dictionary from god config
        takes fmri from any region and makes it a 3d array in the shape of a cube that would fit the whole VC
        this is done for all trials.
        """
        voxel_x = subject1.get_metadata('voxel_x', where=area[roi])#might be smaller than full size for other regions
        voxel_y = subject1.get_metadata('voxel_y', where=area[roi])
        voxel_z = subject1.get_metadata('voxel_z', where=area[roi])
        fmri = subject1.select(rois[roi])
        pc=np.zeros((fmri.shape[0],full_x_shape,full_y_shape,full_z_shape))
        for i in tqdm(range(len(voxel_x))):
            x=voxel_x[i]
            y=voxel_y[i]
            z=voxel_z[i]
            x_ind = int((x - full_voxel_x.min())/3)
            z_ind = int((z - full_voxel_z.min())/3)
            y_ind = int((y - full_voxel_y.min())/3)
            for n in range(fmri.shape[0]):
                pc[n][x_ind][y_ind][z_ind]=fmri[n][i]

        return(pc)

    for roi in {'VC' : 'ROI_VC = 1'}: #for now only VC later on use rois dictionary from god_config
        co = coordinates(roi)
        np.save(dir_path+'/data/'+subject+'_'+roi+'_'+'coordinates',co)

        fmri_padded=padded_fmri(roi)
        np.save(dir_path+'/data/'+subject+'_'+roi+'_'+'fmri',fmri_padded)
