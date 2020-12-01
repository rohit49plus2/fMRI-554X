import numpy as np
import os
from god_config import *
import bdpy
import tqdm
dir_path = os.path.dirname(os.path.realpath(__file__)) #current directory
import matplotlib.pyplot as plt
for subject in {'Subject1' : dir_path+'/original code/data/Subject1.h5'}: #for now only subject1, later on replace with subjects dictionary from god_config
    #Load subject's data
    roi='VC'
    subject1=bdpy.BData(subjects[subject])
    # del X
    voxel_x = subject1.get_metadata('voxel_x', where=area[roi])
    voxel_y = subject1.get_metadata('voxel_y', where=area[roi])
    voxel_z = subject1.get_metadata('voxel_z', where=area[roi])
    datatype = subject1.select('DataType')   # Data type
    i_train = (datatype == 1).flatten()    # Index for training 1200 trials
    i_test_pt = (datatype == 2).flatten()  # Index for perception test 35 runs of 50 images = 1750
    i_test_im = (datatype == 3).flatten()

    fmri_seen=np.mean(subject1.select(rois[roi])[i_train+i_test_pt][:],axis=0)
    fmri_imagined=np.mean(subject1.select(rois[roi])[i_test_im][:],axis=0)

    fmri=np.fft.fftn(fmri_seen)
    fmri2=np.fft.fftn(fmri_imagined)
    fmri=np.log(np.abs(np.fft.fftshift(fmri))**2)
    fmri2=np.log(np.abs(np.fft.fftshift(fmri2))**2)
    x_shape=int((voxel_x.max()-voxel_x.min())/3+1)
    y_shape=int((voxel_y.max()-voxel_y.min())/3+1)
    z_shape=int((voxel_z.max()-voxel_z.min())/3+1)
    # pc=np.zeros((x_shape,y_shape,z_shape))
    # for i in range(len(voxel_x)):
    #     x=voxel_x[i]
    #     y=voxel_y[i]
    #     z=voxel_z[i]
    #     x_ind = int((x - voxel_x.min())/3)
    #     z_ind = int((z - voxel_z.min())/3)
    #     y_ind = int((y - voxel_y.min())/3)
    #     pc[x_ind][y_ind][z_ind]=np.abs(fmri[i])
    #
    # pc2=np.zeros((x_shape,y_shape,z_shape))
    # for i in range(len(voxel_x)):
    #     x=voxel_x[i]
    #     y=voxel_y[i]
    #     z=voxel_z[i]
    #     x_ind = int((x - voxel_x.min())/3)
    #     z_ind = int((z - voxel_z.min())/3)
    #     y_ind = int((y - voxel_y.min())/3)
    #     pc2[x_ind][y_ind][z_ind]=np.abs(fmri2[i])

    # x= np.arange(voxel_x.min(), voxel_x.max()+3, 3)
    # y= np.arange(voxel_y.min(), voxel_y.max()+3, 3)
    # z= np.arange(voxel_z.min(), voxel_z.max()+3, 3)
    # print(x.shape)
    # print(y.shape)
    # print(z.shape)
    x= np.fft.fftfreq(x_shape,d=1/3)
    y= np.fft.fftfreq(y_shape,d=1/3)
    z= np.fft.fftfreq(z_shape,d=1/3)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    xx, yy ,zz = np.meshgrid(x, y, z,indexing='ij')

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(1, 2, 1,projection="3d")
    # plot=ax.scatter(xx.flatten(),yy.flatten(),zz.flatten(),s=20,c=pc,cmap='coolwarm',vmin=-20,vmax=+20)
    # ax.set_xlabel('X Axes')
    # ax.set_ylabel('Y Axes')
    # ax.set_zlabel('Z Axes')
    # ax.set_title("Average Logarithm of Absolute value of FFT fmri \nduring seen experiments in " + roi + " of " + subject,fontsize=16)
    # fig.colorbar(plot,shrink=0.5)
    #
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # plot=ax.scatter(xx.flatten(),yy.flatten(),zz.flatten(),s=20,c=pc2,cmap='coolwarm',vmin=-20,vmax=+20)
    # ax.set_xlabel('X Axes')
    # ax.set_ylabel('Y Axes')
    # ax.set_zlabel('Z Axes')
    # ax.set_title("Average Logarithm of Absolute value of FFt fmri \nduring imagined experiments in " + roi + " of " + subject,fontsize=16)
    # fig.colorbar(plot,shrink=0.5)
    # plt.show()



    all_fmri=subject1.select(rois[roi])
    X=[]
    Y=[]
    for i in range(len(datatype)):
        print(i)
        if datatype[i]==3:
            Y.append(1)
        else:
            Y.append(0)
        fmri=np.fft.fftn(all_fmri[i])
        fmri=np.log(np.abs(np.fft.fftshift(fmri))**2)
        pc=np.zeros((x_shape,y_shape,z_shape))
        for i in range(len(voxel_x)):
            x=voxel_x[i]
            y=voxel_y[i]
            z=voxel_z[i]
            x_ind = int((x - voxel_x.min())/3)
            z_ind = int((z - voxel_z.min())/3)
            y_ind = int((y - voxel_y.min())/3)
            pc[x_ind][y_ind][z_ind]=np.abs(fmri[i])
        X.append(pc)
    X=np.stack(X)
    Y=np.array(Y)
    absavgdata = X
    seen = absavgdata[Y==0]
    imagined = absavgdata[Y==1]
    pc=np.average(seen,axis=0)
    pc2=np.average(imagined,axis=0)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1,projection="3d")
    plot=ax.scatter(xx.flatten(),yy.flatten(),zz.flatten(),s=20,c=pc,cmap='coolwarm',vmin=-20,vmax=+20)
    ax.set_xlabel('X Axes')
    ax.set_ylabel('Y Axes')
    ax.set_zlabel('Z Axes')
    ax.set_title("Average Logarithm of Absolute value of FFT fmri \nduring seen experiments in " + roi + " of " + subject,fontsize=16)
    fig.colorbar(plot,shrink=0.5)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    plot=ax.scatter(xx.flatten(),yy.flatten(),zz.flatten(),s=20,c=pc2,cmap='coolwarm',vmin=-20,vmax=+20)
    ax.set_xlabel('X Axes')
    ax.set_ylabel('Y Axes')
    ax.set_zlabel('Z Axes')
    ax.set_title("Average Logarithm of Absolute value of FFt fmri \nduring imagined experiments in " + roi + " of " + subject,fontsize=16)
    fig.colorbar(plot,shrink=0.5)
    plt.show()

    absavgdata=np.zeros(X.shape[0])
    for i in range(absavgdata.shape[0]):
        absavgdata[i]=np.mean(X[i])
    seen = absavgdata[Y==0]
    imagined = absavgdata[Y==1]

    bins = np.linspace(0,2.5,1000)

    fig,ax1 = plt.subplots()
    ax1.hist(seen, bins, alpha = 0.6, color = 'c', label = "Seen")
    ax2 = ax1.twinx()
    ax2.hist(imagined, bins, alpha = 0.6, color = 'g', label = "Imagined")
    fig.legend(bbox_to_anchor = (0.85,0.8))
    plt.show()
