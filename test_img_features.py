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
# Main #################################################################

#Load subject 1's data
subject1=bdpy.BData(subjects['Subject1'])
image_features=bdpy.BData(image_feature_file)
for roi in rois:
    x=subject1.select(rois[roi])#load fMRI data
    datatype=datatype = subject1.select('DataType')   # Data type
    labels = subject1.select('stimulus_id')  # Image labels in brain data
    print("roi: ", roi)
    print("x shape: ", x.shape)
    print("datatype shape: ", datatype.shape)
    print("labels shape: ", labels.shape)
    i_train = (datatype == 1).flatten()    # Index for training
    i_test_pt = (datatype == 2).flatten()  # Index for perception test
    i_test_im = (datatype == 3).flatten()  # Index for imagery test
    print("indexes for imagery:", i_test_im)
    print("labels unique",np.unique(labels).shape)
    print("i_train sum",i_train.sum())
    print("i_test_pt sum",i_test_pt.sum())
    print("i_test_im sum",i_test_im.sum())
