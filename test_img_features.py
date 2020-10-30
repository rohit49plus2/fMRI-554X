# Test code to see how the data files are stored
# analysis_FeaturePrediction.py in the /original code folder is a good way to understand

from bdpy import BData
from bdpy.util import makedir_ifnot, get_refdata
import numpy as np
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
bdata=BData(dir_path+'/original code/data/ImageFeatures.h5')
# Show 'key' and 'description' of metadata
bdata.show_metadata()

cnn_8=bdata.select('cnn8')
y_label = bdata.select('ImageID')
print(cnn_8.shape)
print(y_label.shape)
