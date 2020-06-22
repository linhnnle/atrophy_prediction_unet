import os
import numpy as np
import nibabel as nib
from sklearn.metrics import auc, precision_score, recall_score, roc_curve


def dice_score(y_true, y_pred, smooth=0.0000001, threshold_true=0.1, threshold_pred=0.5):
    y_true_f = y_true.flatten() >= threshold_true
    y_pred_f = y_pred.flatten() >= threshold_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def specificity(y_true, y_pred, smooth = 0.00001, threshold_true = 0.1, threshold_pred =0.5):
    y_neg_f = y_true.flatten() < threshold_true
    y_pred_pos_f = y_pred.flatten() >= threshold_pred
    false_pos = np.sum(y_neg_f * y_pred_pos_f)
    return np.sum(y_neg_f) / (np.sum(y_neg_f) + false_pos + smooth)
def vol_diff(y_true, y_pred, threshold_true = 0.1, threshold_pred =0.5):
    y_true_f = y_true.flatten() >= threshold_true
    y_pred_f = y_pred.flatten() >= threshold_pred
    return np.sum(y_pred_f) - np.sum(y_true_f)
def vol_pred(y_pred,threshold_pred=0.5):
    y_pred_f = y_pred.flatten() >= threshold_pred
    return np.sum(y_pred_f)
def weighted_dice(y_true,y_pred,smooth = 0.00001,threshold_true = 0.1, threshold_pred =0.5):
    y_true_f = y_true.flatten() >= threshold_true
    y_pred_f = y_pred.flatten() >= threshold_pred
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (1.7 * np.sum(y_true_f) + 0.3 * np.sum(y_pred_f) + smooth)
def get_max_for_each_slice(data):
    '''
    data has to be 3d
    '''
    assert len(data.shape) == 3 , 'input data is not 3d'
    max_list = []
    for slice_num in range(data.shape[2]):
        max_list.append(np.max(data[:,:,slice_num]))
    return max_list

def define_laterality(data,threshold):
    midline = int(data.shape[0] / 2)
    max_list_gt = get_max_for_each_slice(data)
    lesion_left = 0
    lesion_right = 0
    lesion_side = ''
    for slice_num in range(len(max_list_gt)):
        if max_list_gt[slice_num] > threshold:
            if np.sum(data[:midline, :, slice_num]) > np.sum(data[midline:, :,
                                                                slice_num]):  ## If stroke in Left side of the image and Right side of the brain
                lesion_left += 1
            elif np.sum(data[:midline, :, slice_num]) < np.sum(data[midline:, :,
                                                                  slice_num]):  ## If stroke in Right side of the image and Left side of the brain
                lesion_right += 1
    if (lesion_left > lesion_right and (lesion_right > 3)) or (lesion_left < lesion_right and (lesion_left > 3)):
        lesion_side = 'B'
    elif lesion_left > lesion_right:
        lesion_side = 'L'
    elif lesion_right > lesion_left:
        lesion_side = 'R'
    # print(lesion_left,lesion_right)
    return lesion_side

def metrics_output(y_true, y_pred,threshold_true,threshold_pred):
    '''
    output all the metrics including auc dice recall precision f1score and volume difference
    '''
    fpr, tpr, thresholds = roc_curve(y_true>threshold_true,y_pred)
    auc_hemisphere = auc(fpr, tpr)
    precision = precision_score(y_true>threshold_true, y_pred>threshold_pred)
    recall = recall_score(y_true>threshold_true, y_pred>threshold_pred)
    dice = dice_score(y_true,y_pred, threshold_true=threshold_true,threshold_pred=threshold_pred)
    spec = specificity(y_true, y_pred,threshold_true=threshold_true,threshold_pred=threshold_pred)
    voldiff = 0.008*vol_diff(y_true, y_pred,threshold_true=threshold_true,threshold_pred=threshold_pred)
    volpred = 0.008*vol_pred(y_pred,threshold_pred)
    f1score = 2 * precision * recall / (precision + recall + 0.0001)
    return auc_hemisphere, precision, recall, dice, spec, voldiff, volpred, f1score, fpr, tpr, thresholds

def load_nii(path):
  data = nib.load(path)
  output = data.get_fdata()
  output = np.maximum(0, np.nan_to_num(output, 0))
  return output

def stroke_test_metrics(dir_result = '/data/yuanxie/Enhao/stroke_cv',dir_stroke = '/data/yuanxie/stroke_preprocess173/', dir_brainmask = '/Users/admin/controls_stroke_DL/001/inputs_aug0.hdf5',
                subj_list = ['30077A'], model_name = 'test_model',mask_contrast = 0, threshold_true =0.5, threshold_pred = 0.5,printout=True,upper_lim=91,lower_lim=0,savedata=True,dwimask=True):
    '''

    :param dir_result: where to find predicted h5 file
    :param dir_stroke: where to find input h5 file
    :param dir_brainmask: where to find T1 template h5 file
    :param subj_list: list of the test cases
    :param model_name: model name
    :param mask_contrast: contrast for PWI masking. 4 = MTT, 5 = Tmax, corresponding to the input.h5 4th dimension.
    :param threshold_true: when ground truth > threshold_true, count as positive
    :param threshold_pred: when prediction > threshold_pred, count as positive
    :return: the list_result containing auc, precision, recall , dice, auc all.
    '''
    '''
    setup gpu
    '''
    list_subject = sorted(subj_list)
    # print(list_subject)

    list_result = {'subject':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'abs_volume_difference':[]}
    all_y_true = np.array([])
    all_y_pred = np.array([])
    for subject in list_subject:

        # load nifti
        path_gt = dir_stroke + "{}/LESION.nii".format(subject)

        if not os.path.exists(path_gt):
            print('no ground truth file for', subject)
            continue

        data_gt = load_nii(path_gt)

        name_output = 'prediction_' + model_name + '_{0}'.format(subject)  + '.nii'
        path_output = os.path.join(dir_result, name_output)
        data_output = load_nii(path_output)
        # input brain masking from T1 template.
        T1temp = load_nii(dir_brainmask)
        T1temp = T1temp[:, :, lower_lim:upper_lim]

        if mask_contrast == 0:
            mask_name = 'DWI.nii'

        if mask_contrast < 9:
            path_PWImask = dir_stroke + '{0}/'.format(subject) + mask_name
            data_PWImask = load_nii(path_PWImask)
            data_PWImask = data_PWImask[:, :, lower_lim:upper_lim]

        #get DWI mask
        if dwimask:
            path_DWI = dir_stroke + '{0}/'.format(subject) + 'DWI.nii'
            dwi = load_nii(path_DWI)
            dwi = dwi[:, :, lower_lim:upper_lim] * (T1temp > 0)
            mean_dwi = np.mean(dwi[np.nonzero(dwi)])

        # compute auc
        y_true_data = []
        y_pred_data = []
        midline = int(data_gt.shape[0]/2)
        lesion_side = define_laterality(data_gt,threshold_true)

        if lesion_side != 'B':
            if lesion_side == 'L': ## If stroke in Left side of the image and Right side of the brain
                T1temp[midline:, :, :] = 0
            elif lesion_side == 'R': ## If stroke in Right side of the image and Left side of the brain
                T1temp[:midline, :, :] = 0
            else:
                print('check code and data. Left lesion  = Right lesion ')
        for index in range(data_gt.shape[2]):
            brain_mask = (T1temp[:, :, index] > 0) * 1.
            if mask_contrast < 9:
                PWImask = np.maximum(0, np.nan_to_num(data_PWImask[:, :, index], 0))
                brain_mask = brain_mask * (PWImask >0)
            if dwimask:
                brain_mask = brain_mask * (dwi[:, :, index] > (0.3 * mean_dwi))
            #### calculate AUC based on hemisphere!
            brain_mask[brain_mask == 0] = np.NaN  # 0 still be calculated in later steps, so convert to NaN
            y_true_masked = data_gt[:, :, index] * brain_mask
            y_true_data.append(y_true_masked)
            y_pred_masked = data_output[:, :, index] * brain_mask
            y_pred_data.append(y_pred_masked)

        y_true = np.array(y_true_data).flatten()
        y_true = y_true[~np.isnan(y_true)]
        y_pred = np.array(y_pred_data).flatten()
        y_pred = y_pred[~np.isnan(y_pred)]
        auc_hemisphere, precision, recall, dice, spec, voldiff, volpred, _, _, _ ,_= metrics_output(y_true, y_pred, threshold_true, threshold_pred)
        if printout:
            print(subject, dice, auc_hemisphere, precision, recall, spec, voldiff, volpred, abs(voldiff),f1score)
        if savedata:
            label = (y_true > threshold_true) * 1.
            pred = y_pred
            id = range(len(y_true))
            output_csv = np.column_stack((id,label,pred))
            np.savetxt(dir_result + subject + '.csv', output_csv, fmt='%1.f,%1.f,%4.5f',delimiter=",")
        list_result['subject'].append(subject)
        list_result['auc'].append(auc_hemisphere)
        list_result['precision'].append(precision)
        list_result['recall'].append(recall)
        list_result['dice'].append(dice)
        list_result['specificity'].append(spec)
        list_result['volume_difference'].append(voldiff)
        list_result['volume_predicted'].append(volpred)
        list_result['abs_volume_difference'].append(abs(voldiff))
        # list_result['f1_score'].append(f1score)

        all_y_true = np.append(all_y_true,y_true)
        all_y_pred = np.append(all_y_pred,y_pred)

    if savedata:
        label_all = (all_y_true > threshold_true) * 1.
        output_csv_all = np.column_stack((label_all, all_y_pred))
        np.savetxt(dir_result + 'all_pred.csv', output_csv_all,fmt='%1.f,%4.5f',delimiter=',')

    all_auc_hemisphere, all_precision, all_recall, all_dice, all_spec, all_voldiff, all_volpred, all_f1score,fpr,tpr,thresholds = metrics_output(all_y_true, all_y_pred, threshold_true, threshold_pred)
    list_result_all = {'auc': all_auc_hemisphere, 'precision': all_precision, 'recall': all_recall,
                     'specificity': all_spec, 'dice': all_dice, 'volume_difference': all_voldiff,
                     'volume_predicted': all_volpred,
                     'abs_volume_difference': abs(all_voldiff)}
    return list_result, list_result_all,fpr,tpr,thresholds
