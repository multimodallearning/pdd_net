import numpy as np
import nibabel as nib
import struct
from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage.interpolation import map_coordinates as map_coordinates
#import torch
#import torch.nn as nn
#import torch.nn.functional as F


import argparse



def main():

    parser = argparse.ArgumentParser()
    #inputdatagroup = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--input_field", dest="input_field", help="input pdd displacement field (.npz) half resolution", default=None, required=True)
    parser.add_argument("--input_moving", dest="input_moving",  help="input moving scan (.nii.gz)", default=None, required=True)
    parser.add_argument("--output_warped", dest="output_warped",  help="output waroed scan (.nii.gz)", default=None, required=True)


    options = parser.parse_args()
    d_options = vars(options)
    
    
    input_field = np.load(d_options['input_field'])['arr_0']
    _, H1, W1, D1 = input_field.shape
    H = int(H1*2); W = int(W1*2); D = int(D1*2);
    identity = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    
    disp_field = np.zeros((3,H,W,D)).astype('float32')
    disp_field[0] = zoom(input_field[0].astype('float32'),2,order=2)
    disp_field[1] = zoom(input_field[1].astype('float32'),2,order=2)
    disp_field[2] = zoom(input_field[2].astype('float32'),2,order=2)
    moving = nib.load(d_options['input_moving']).get_fdata()
    moving_warped = map_coordinates(moving, identity + disp_field, order=0) #assuming a segmentation -> nearest neighbour interpolation

    nib.save(nib.Nifti1Image(moving_warped,np.eye(4)),d_options['output_warped'])




if __name__ == '__main__':
    main()
