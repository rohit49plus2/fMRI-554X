import numpy as np
import os
import god_config as config
import bdpy
from tqdm import tqdm
dir_path = os.getcwd()

def pad_data(roi):
    voxels_x = []
    voxels_y = []
    voxels_z = []
    voxel_width_x = 0
    voxel_width_y = 0
    voxel_width_z = 0
    min_voxel_x = 0
    min_voxel_y = 0
    min_voxel_z = 0
    for i,subject in enumerate(config.subjects):
        voxels_x.append(bdpy.BData(config.subjects[subject]).get_metadata('voxel_x', where=config.area[roi]))
        voxels_y.append(bdpy.BData(config.subjects[subject]).get_metadata('voxel_y', where=config.area[roi]))
        voxels_z.append(bdpy.BData(config.subjects[subject]).get_metadata('voxel_z', where=config.area[roi]))

        #Take the width of the padded data to be the largest width of the data from each subject
        #Note: The spatial difference in each voxel is 3mm, hence the 3 in the calculation to
        #      convert from spatial data to array indicies
        width_x = int((voxels_x[i].max()-voxels_x[i].min())/3+1)
        width_y = int((voxels_y[i].max()-voxels_y[i].min())/3+1)
        width_z = int((voxels_z[i].max()-voxels_z[i].min())/3+1)
        if width_x > voxel_width_x:
            voxel_width_x = width_x
        if width_y > voxel_width_y:
            voxel_width_y = width_y
        if width_z > voxel_width_z:
            voxel_width_z = width_z
        
        #Take the minimum voxel to be the minimum of all subjects
        min_x = voxels_x[i].min()
        min_y = voxels_y[i].min()
        min_z = voxels_z[i].min()
        if min_x < min_voxel_x:
            min_voxel_x = min_x
        if min_y < min_voxel_y:
            min_voxel_y = min_y
        if min_z < min_voxel_z:
            min_voxel_z = min_z

    print ("Minimum voxels locations:\t{}\t{}\t{}".format(min_voxel_x, min_voxel_y, min_voxel_z))
    print ("Largest voxel widths:\t\t{}\t{}\t{}".format(voxel_width_x, voxel_width_y, voxel_width_z))
    padded_data = np.array([])
    for k,subject in enumerate(tqdm(config.subjects)):
        fmris = bdpy.BData(config.subjects[subject]).select(config.rois[roi])
        subject_data = np.zeros((fmris.shape[0], voxel_width_x, voxel_width_y, voxel_width_z))
        
        #Not only do each of the subjects have different voxel widths, they are also offset
        #So subject1 might go from -42 to 40 with a width of 82, whereas subject2 goes from
        #-36 to 38 with a width of 74, but subject1 is offset to the left and subject2 is 
        #offset to the right. We will try to center the fmris of each subject
        
        #The difference in widths
        x_width_dif = voxel_width_x - int((voxels_x[k].max()-voxels_x[k].min())/3+1)
        y_width_dif = voxel_width_y - int((voxels_y[k].max()-voxels_y[k].min())/3+1)
        z_width_dif = voxel_width_z - int((voxels_z[k].max()-voxels_z[k].min())/3+1)
        
        #The current index offset
        x_min_dif = (voxels_x[k].min() - min_voxel_x)//3+1
        y_min_dif = (voxels_y[k].min() - min_voxel_y)//3+1
        z_min_dif = (voxels_z[k].min() - min_voxel_z)//3+1
        
        #The correct offest is half the difference in width. We can get a correction by 
        #taking the difference between the correct offset and the current offset
        correction_x = x_width_dif//2 - x_min_dif
        correction_y = y_width_dif//2 - y_min_dif
        correction_z = z_width_dif//2 - z_min_dif
        
        #print (z_width_dif, z_min_dif, correction_z, min_voxel_z, voxels_z[k].min())
        for i in range(fmris.shape[0]):
            for j in range(len(voxels_x[k])):
                ind_x = int((voxels_x[k][j] - min_voxel_x)/3 + correction_x)
                ind_y = int((voxels_y[k][j] - min_voxel_y)/3 + correction_y)
                ind_z = int((voxels_z[k][j] - min_voxel_z)/3 + correction_z)
                subject_data[i][ind_x][ind_y][ind_z] = fmris[i][j]
                 
        padded_data = np.append(padded_data,subject_data)
    padded_data = padded_data.reshape((-1,voxel_width_x,voxel_width_y,voxel_width_z))
    return padded_data

if __name__ == "__main__":
    for roi in config.rois:
        print ("Padding ROI:",roi,"for all subjects") 
        datatype = np.array([])
        for subject in config.subjects:
            subject_data = bdpy.BData(config.subjects[subject]) 
            datatype = np.append(datatype,subject_data.select('DataType'))
            np.save(dir_path+'/padded_data/'+'datatype',datatype)
            padded_data = pad_data(roi)
            np.save(dir_path+'/padded_data/'+roi, padded_data)
#    padded_data = pad_data('VC')
#    np.save(dir_path+'/padded_data/VC', padded_data)
                  
                  
                  
