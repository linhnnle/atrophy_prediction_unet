
''' basic dependencies '''
import numpy as np
import os
import json
import datetime
from time import time
import pydicom
import nibabel as nib
import h5py
import pdb

from keras.models import load_model
from subtle_fileio import *
from subtle_metrics import *
from subtle_generator import DataGenerator

def dice_score(y_true, y_pred, smooth=0.0000001,threshold_true=0.1, threshold_pred=0.5):
    y_true_f = y_true.flatten()>=threshold_true
    y_pred_f = y_pred.flatten()>=threshold_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def export_to_h5(data_export, path_export, key_h5='init', dtype=np.float32):
    """Export numpy array to h5.

    Parameters
    ----------
    data_export (numpy array) - data to export.
    path_export (str) - path to h5 file.
    key_h5 - key for h5 file.

    """
    with h5py.File(path_export,'w') as f:
        f.create_dataset(key_h5, data=data_export.astype(dtype))

    print('H5 exported to: {}'.format(path_export))

def stroke_test(dir_stroke = '/data/yuanxie/stroke_preprocess173/', dir_source = '/data/yuanxie/deepstroke173/',
                subj_list = ['30077A'], model_name = 'test_model', output_gt = True, output_flair = True,
                 dir_ckpt = '/Users/admin/stroke_DL/ckpt_stroke',filename_checkpoint = 'model_stroke_test.ckpt',
                 filename_model='model_stroke_test.json', output_path = '../stroke_cv/',followup_image_name='FLAIR',
                 shape_px_width = 128, shape_px_height =128, num_contrast_input = 7, num_channel_output = 1,
                 output_range_low = 0., out_put_range_high = 1.0, with_batch_norm = True, batch_size = 16,
                 num_conv_per_pooling = 3, num_poolings = 3, num_workers = 16, num_slice_expand = 2,
                keras_memory=0.4, lower_lim=0,upper_lim=91):
    ''''''
    '''
    setup gpu
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    max_queue_size = num_workers*2

    keras_backend = 'tf'
    y_range = [output_range_low,out_put_range_high]
    # dimension
    num_slice_25d = num_slice_expand*2+1
    index_slice_mid = num_slice_expand
    num_channel_input = num_slice_25d*num_contrast_input
    shape_resize = [shape_px_width,shape_px_height]
    img_rows = shape_resize[0]
    img_cols = shape_resize[1]
    # generator settings
    generator_resize_data = False
    generator_normalize_data = False
    generator_mask_data = False
    generator_sanitize_data = False

    generator_axis_slice = 0
    generator_split_input = True
    with_multi_lateral_filter = False


    '''
    load data
    '''

    num_subject = len(subj_list)
    print(subj_list)
    print('Load model in path:', dir_ckpt, filename_checkpoint)
    filepath_ckpt = os.path.join(dir_ckpt, filename_checkpoint)

    model = load_model(filepath_ckpt[:-5]+'.h5',custom_objects={'test_loss': test_loss,# change if loss function changes
                       'dice_coef': dice_coef,'dice_coef_loss':dice_coef_loss,
                       'seg_crossentropy':seg_crossentropy,'precision':precision,
                       'recall':recall,'f1_score':f1_score,'l1_loss':l1_loss,
                       'vol_diff':vol_diff,'weighted_dice':weighted_dice,'weighted_dice_loss':weighted_dice_loss,
                       'sce_dice_l2_vol_loss':sce_dice_l2_vol_loss, 'weighted_ce_l1_bycase':weighted_ce_l1_bycase,
                       'seg_crossentropy_weighted_bycase':seg_crossentropy_weighted_bycase,'ssim_loss':ssim_loss}) ## load the final model (h5 format),2018/11/19 written by Charles.
    model.load_weights(filepath_ckpt) # load the best performance, based on best validation?? may not correctly load reported by Charles


    '''
    load model
    '''

    vol_output = []
    vol_gt = []
    for index_data in range(0, num_subject):
        subject_name = subj_list[index_data]  # '30001A'
        dir_subject = os.path.join(dir_source, subject_name)

        # load input
        try:
            print(subject_name)

            input_file = os.path.join(dir_stroke, subject_name, 'inputs_aug0.hdf5')
            inputs = load_h5(input_file, transpose_dims=[0, 1, 2, 3])
            gt_file = os.path.join(dir_stroke, subject_name, 'output_aug0.hdf5')
            gt = load_h5(gt_file, transpose_dims=[0, 1, 2, 3])
        except:
            print('error in loading h5')
            continue

        # get slices
        num_slices = gt.shape[0]
        list_samples_test = [[[input_file], gt_file, x, 1] for x in
                             range(num_slice_expand, num_slices - num_slice_expand)]

        num_samples_per_subj = len(list_samples_test)
        # predict with generator
        params_generator = {'dim_x': img_rows,
                            'dim_y': img_cols,
                            'dim_z': num_channel_input,
                            'dim_25d': num_slice_25d,
                            'dim_output': 1,
                            'batch_size': num_samples_per_subj,
                            'shuffle': False,
                            'verbose': 1,
                            'scale_data': 1,
                            'scale_baseline': 0.0,
                            'normalize_per_sample': False,
                            'para': {
                                'num_contrast_input': num_contrast_input
                            },
                            'normalize_data': generator_normalize_data,
                            'resize_data': generator_resize_data,
                            'mask_data': generator_mask_data,
                            'sanitize_data': generator_sanitize_data,
                            'axis_slice': generator_axis_slice
                            }

        # get samples parameters
        test_generator = DataGenerator(**params_generator).generate([], list_samples_test)
        # get number of epoch
        num_step = ((len(list_samples_test) - num_slice_expand * 2) - 1) // batch_size + 1

        '''
        predict
        '''

        num_steps=1

        output = model.predict_generator(
            generator=test_generator,
            steps=num_steps,  # steps_per_epoch = steps_per_epoch,
            max_queue_size=max_queue_size,
            workers=num_workers,
            use_multiprocessing=False
        )

        print('Number of samples: ',num_samples_per_subj)
        print('Number of subjects to predict: ',num_subject)
        flair_dir_source = dir_source #This is the directory where the original nii.gz files are located as defined by dict_source



        flair_path = os.path.join(flair_dir_source, subj_list[index_data]) + '/{}.nii'.format(followup_image_name)
        proxy = nib.load(flair_path)

        lesion_path = os.path.join(flair_dir_source, subj_list[index_data]) + '/LESION.nii'
        proxy1 = nib.load(lesion_path)


        flair_array = np.asarray(proxy.dataobj)
        flair_pad_reshaped = data_reshape(flair_array,[shape_px_height,shape_px_width])
        flair_pad = flair_pad_reshaped[:,:,lower_lim:upper_lim]
        lesion_array = np.asarray(proxy1.dataobj)




        export_name = 'prediction_'+model_name+'_'+subj_list[index_data]+'.nii'
        new_data = np.squeeze(output)
        new_data_trans = new_data.transpose(1,2,0)
        left_pad_dif = int(np.floor((flair_pad.shape[2]-new_data_trans.shape[2])/2))
        right_pad_dif = int(np.ceil((flair_pad.shape[2]-new_data_trans.shape[2])/2))
        new_data_trans = np.pad(new_data_trans,((0,0),(0,0),(left_pad_dif,right_pad_dif)),'constant',constant_values = 0)
        ## added 11/26 by Yannan to reshape the image
        new_data_trans = data_reshape(new_data_trans, [91,109]).astype(np.float32)
        ##
        new_image = nib.Nifti1Image(new_data_trans, proxy.affine)
        nib.save(new_image, os.path.join(output_path, export_name))
        print('prediction image shape: ',new_data_trans.shape)

        if output_flair:
            export_name1 = 'flair_'+model_name+'_'+subj_list[index_data]+'.nii'
            ## added 11/26 by Yannan to reshape the image
            flair_pad = data_reshape(flair_pad, [91,109]).astype(np.float32)
            ##
            new_image1 = nib.Nifti1Image(flair_pad, proxy.affine)
            nib.save(new_image1, os.path.join(output_path, export_name1))
            print('base flair shape: ',flair_pad.shape)
        if output_gt:
            export_name2 = 'gt_'+model_name+'_'+subj_list[index_data]+'.nii'
            new_data_gt = np.squeeze(gt)
            new_data_gt_trans = new_data_gt.transpose(1,2,0)
            left_pad_dif = int(np.floor((flair_pad.shape[2]-new_data_gt_trans.shape[2])/2))
            right_pad_dif = int(np.ceil((flair_pad.shape[2]-new_data_gt_trans.shape[2])/2))
            new_data_gt_trans = np.pad(new_data_gt_trans,((0,0),(0,0),(left_pad_dif,right_pad_dif)),'constant',constant_values = 0)
            ## added 11/26 by Yannan to reshape the image
            new_data_gt_trans = data_reshape(new_data_gt_trans, [91,109]).astype(np.float32)
            ##
            new_image_gt = nib.Nifti1Image(new_data_gt_trans, proxy.affine)
            nib.save(new_image_gt, os.path.join(output_path, export_name2))
            print('ground truth lesion shape: ',new_image_gt.shape)


        print(['nifti outputs for subject ',index_data+1,' done!'])
