
import os
from os import path
import random
from stroke_train_function import stroke_train_img_segmentation
from stroke_test_function_with_predictions import stroke_test
from stroke_testdice import stroke_test_metrics
from create_fig_for_model import *

fold1 = ['subject1'] #subfolder name (usually the patient ID) that contains data
fold2 = ['subject2']
fold3 = ['subject3']
fold4 = ['subject4']
fold5 = ['subject5']
list_all = fold1,fold2,fold3,fold4,fold5
# main input to function, change if needed
dir_of_train = 'data/train/' # data folder that contains training data
dir_of_test = 'data/train/'# data folder that contains testing data, since we are using 5-fold crossvalidation, its the same folder here.
dir_source = 'data/source/'# source image that used to overlap the model output and calculate AUC / dice
ext_data = 'hdf5'
log_dir = 'log/'
dir_ckpt = '../ckpt_stroke'
num_epochs= 20
num_contrast_input = 3
mask_contrast = 0 #default is 0, use DWI as mask
gpu = '1' # setup which GPU you want to use
lower_lim = 0 #the limit tells the test function to output only part of the image. usually corresponding to the dimension in preprocessed h5 files. lower limit for the slices to include in an image volume
upper_lim = 60 # upper limit of the image volume
dir_brainmask = 'data/brain_mask.nii' # brain masked use in the test to calculate AUC/dice etc.
batch_size = 16
lr_init = 0.0005
num_conv_per_pooling = 2
num_poolings = 3
num_of_aug = 2 # 1=not include mirrored image, 2=include mirrored image
model_select = 'dropout'
model_name_ori = 'test_model' #change every time you train a new model.
output_path = model_name_ori + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)


loss_mode = 'bycase'

for cv in range(0,5):
    subj_list_train = [list_all[(cv+x)%5] for x in range(2,5)]
    val_list = list_all[(cv+1)%5]
    subj_list_test = list_all[cv]
    flat_train_subj = []
    for sublist in subj_list_train:
        for item in sublist:
            flat_train_subj.append(item)
    model_name = model_name_ori + '_fold{0}'.format(cv+1)
    filename_checkpoint = 'model_stroke_' + model_name + '.hdf5'
    filename_model = 'model_stroke_' + model_name + '.json'
    # training
    # stroke_train_img_segmentation(dir_of_train, flat_train_subj, lr_init=lr_init, loss_mode = loss_mode,gpu = gpu, model_select = model_select, val_list = val_list, extension_data = ext_data, log_dir=log_dir,dir_ckpt=dir_ckpt,filename_checkpoint=filename_checkpoint,filename_model=filename_model,num_epochs=num_epochs,num_contrast_input = num_contrast_input,batch_size=batch_size,num_of_aug=num_of_aug, num_conv_per_pooling=num_conv_per_pooling,num_poolings=num_poolings)
    # generate output for test cases
    # stroke_test(dir_stroke=dir_of_test, dir_source=dir_source, subj_list=subj_list_test, model_name=model_name,dir_ckpt=dir_ckpt,filename_model = filename_model,filename_checkpoint=filename_checkpoint,num_contrast_input = num_contrast_input,output_path = output_path, lower_lim = lower_lim, upper_lim = upper_lim)

#
threshold_true = 0.5 # thresholding has been done in preprocess. if preprocess is already 0.9, try low value here.
rangelist = [0.4,0.5,0.6] # you can test difference thresholds in the output and generate dice and other metrics

median_metrics = {'thres':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'abs_volume_difference':[]}
for thres in rangelist:
    print(model_name_ori)
    print('true:',threshold_true,'prediction',thres)
    summary_list = {'subject':[],'auc':[],'precision':[],'recall':[],'specificity':[],'dice':[],'volume_difference':[],'volume_predicted':[],'abs_volume_difference':[]}
    datawrite = False
    for cv in range(0,1):
        if thres == 0.5:
            datawrite=True
        subj_list_test = list_all[cv]
        model_name = model_name_ori + '_fold{0}'.format(cv+1)
        result_list, result_list_all,fpr,tpr, thresholds = stroke_test_metrics(printout=False, dir_result = output_path, dir_stroke = dir_source, dir_brainmask = dir_brainmask,subj_list = subj_list_test, model_name = model_name, mask_contrast = mask_contrast, threshold_true =threshold_true, threshold_pred = thres,lower_lim = lower_lim, upper_lim = upper_lim,savedata=datawrite)
        # create roc figure
        if thres == 0.5:
            create_roc(fpr, tpr, result_list_all['auc'], output_path, thresholds,figname='fold{0}_roc.png'.format(cv+1),tablename='fold{0}_roc.h5'.format(cv+1), datawrite=True)
        for key in summary_list:
            summary_list[key] += result_list[key]
    printout_list = ['mean', np.median(summary_list['dice']), np.median(summary_list['auc']), np.median(summary_list['precision']), np.median(summary_list['recall']), np.median(summary_list['specificity']), np.median(summary_list['volume_difference']),np.median(summary_list['volume_predicted'])]

    for key_mean_metrics in median_metrics:
        if key_mean_metrics != 'thres':
            median_metrics[key_mean_metrics] += [[np.percentile(summary_list[key_mean_metrics],25),np.median(summary_list[key_mean_metrics]),np.percentile(summary_list[key_mean_metrics],75)]]
        else:
            median_metrics[key_mean_metrics] += [[thres,thres,thres]]
    print(printout_list)
    save_dict(summary_list,output_path,filename='thres_gt'+str(thres) + '.csv',summary=True)
