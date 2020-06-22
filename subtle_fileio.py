#!/usr/bin/env python
'''
file i/o modules and functions
'''

import pydicom
import nibabel as nib
import numpy as np
import scipy.io as sio
import os
import h5py
import logging
from glob import glob
from scipy.ndimage import zoom

def glob_filename_list(path,searchterm='*',ext=''):
    filename_list = []
    for filename in glob(path+searchterm+ext):
        fn = filename.split('/')
        filename_list.append(fn[-1])
    return filename_list
'''
load data with specific format
'''
def load_h5(path, key_h5='init', transpose_dims=[2, 0, 1, 3]):
    '''Loads h5 file and convert to a standard format.

    Parameters
    ----------
    path: Path to h5 file.

    Return
    ------
    numpy array.
    '''
    with h5py.File(path, 'r') as f:
        data = np.array(f[key_h5])

    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    if transpose_dims is not None:
        data = np.transpose(data, transpose_dims)

    return data

def load_nib(path, transpose_dims=None):
    '''Loads h5 file and convert to a standard format.

    Parameters
    ----------
    path: Path to nifti file.

    Return
    ------
    numpy array.
    '''
    img = nib.load(path)
    data = img.get_data()
    if transpose_dims is not None:
        data = np.transpose(data, transpose_dims)
    return data

'''
get data from certain extension, here we use np or h5 format
'''
def get_data_from_ext(filepath_load, ext_data, return_data=False, return_mean=True):
    data_load = None
    value_mean = None
    if ext_data.startswith('np'):
        data_load = np.load(filepath_load)
        if return_mean:
            value_mean = np.mean(np.abs(data_load))
        if return_data:
            data_load = np.array(data_load)
    elif ext_data == 'h5' or ext_data == 'hdf5':
        data_load = load_h5(filepath_load, key_h5='init',
                            transpose_dims=None)
        value_mean = np.mean(np.abs(data_load))
    elif ext_data.find('nii')>=0:
        data_load = load_nib(filepath_load)
        # modified by Yannan
        if "LESION" in filepath_load:
            np.nan_to_num(data_load, 0)
    # data_load[np.isnan(data_load)] = 0
        value_mean = np.mean(np.abs(data_load))
    else:
        print('not valid extension:'+ext)
    data_shape = data_load.shape
    return data_load, data_shape, value_mean

'''
export file
'''

def export_to_h5(data_export, path_export, key_h5='init', dtype=np.float32, verbose=0):
    '''Export numpy array to h5.

    Parameters
    ----------
    data_export (numpy array): data to export.
    path_export (str): path to h5 file.
    key_h5: key for h5 file.
    '''
    with h5py.File(path_export,'w') as f:
        f.create_dataset(key_h5, data=data_export.astype(dtype))

    logger = logging.getLogger(__name__)
    logger.debug('H5 exported to: {}'.format(path_export))


def export_to_h5(data_export, path_export, key_h5='init', verbose=0):
    with h5py.File(path_export, 'w') as f1:
        f1.create_dataset(key_h5, data=data_export.astype(np.float32))
    if verbose:
        print('updated H5 exported to:', path_export)

'''
process file
'''
def data_reshape(data_load, shape_zoom, axis_slice=2, zoom_order=2):
    data_load_reshape = np.zeros(shape_zoom+[data_load.shape[axis_slice]])
    zoom_factor = [shape_zoom[0]/(data_load.shape[0]+0.0), shape_zoom[1]/(data_load.shape[1]+0.0)]
    # print('zero matrix shape',data_load_reshape.shape,'zoom factor', zoom_factor)
    for i in range(data_load.shape[-1]):
        data_load_reshape[:,:,i] = zoom(np.squeeze(data_load[:,:,i]), zoom_factor, order=zoom_order)

    return data_load_reshape

# 2.5D augment
def mirror_roll(data_load, i_augment):
    nx,ny,nz,nc = data_load.shape
    data_load2 = np.zeros([nx+abs(i_augment)*2,ny,nz,nc])
    data_load2[abs(i_augment):-abs(i_augment),:,:,:] = data_load
    for i in range(0,abs(i_augment)):
        data_load2[i,:,:,:]=data_load[0,:,:,:]
    for i in range(0,abs(i_augment)):
        data_load2[nx+i+abs(i_augment),:,:,:]=data_load[-1,:,:,:]
    data_load_shift = np.roll(np.array(data_load2), i_augment, axis=0)
    return data_load_shift[abs(i_augment):-abs(i_augment),:,:,:]




def resize_data(data, shape_resize=None, zoom_order=2):
    '''Resize data.

    Parameters
    ----------
    data: numpy array, input data of shape [slices, y, x, channel].
    shape_resize: None or length 2 tuple, shape to be resized to.

    Returns
    -------
    data_resize: numpy array.
    '''
    if shape_resize:
        data_resize = np.zeros([data.shape[0], shape_resize[0], shape_resize[1], data.shape[-1]])
        zoom_factor = [shape_resize[0] / data.shape[1], shape_resize[1] / data.shape[2]]
        for i in range(data.shape[0]):
            for j in range(data.shape[-1]):
                data_resize[i, :, :, j] = zoom(data[i, :, :, j], zoom_factor, order=zoom_order)
    else:
        data_resize = data

    return data_resize
