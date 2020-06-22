import tflearn
import numpy as np
from keras import backend as K
# use skimage metrics
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
# psnr with TF
import pdb

from keras.losses import mean_absolute_error, mean_squared_error, binary_crossentropy
import keras_contrib.backend as KC
from keras import backend as K
from keras import models
from keras.applications.vgg16 import VGG16

from tensorflow import log as tf_log
from tensorflow import constant as tf_constant
import tensorflow as tf


# except:
#     print('import keras and tf backend failed')


def ssim_loss(y_true, y_pred):
    kernel = [3, 3]
    k1 = 0.01
    k2 = 0.03
    kernel_size = 3
    max_value = 1.0
    cc1 = (k1 * max_value) ** 2
    cc2 = (k2 * max_value) ** 2
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

    patches_pred = KC.extract_image_patches(
        y_pred, kernel, kernel, 'valid', K.image_data_format())
    patches_true = KC.extract_image_patches(
        y_true, kernel, kernel, 'valid', K.image_data_format())

    bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)

    patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
    patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
    # Get mean
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    # Get covariance
    covar_true_pred = K.mean(
        patches_true * patches_pred, axis=-1) - u_true * u_pred

    ssim = (2 * u_true * u_pred + cc1) * (2 * covar_true_pred + cc2)
    denom = (K.square(u_true) + K.square(u_pred) + cc1) * \
        (var_pred + var_true + cc2)
    ssim /= denom

    return K.mean((1.0 - ssim) / 2.0)
# def ssim_loss(y_true, y_pred):
#     kernel = [3, 3]
#     k1 = 0.01
#     k2 = 0.03
#     kernel_size = 3
#     max_value = 1.0
#     cc1 = (k1 * max_value) ** 2
#     cc2 = (k2 * max_value) ** 2
#     y_true = KC.reshape(y_true, [-1] + list(KC.int_shape(y_pred)[1:]))
#     y_pred = KC.reshape(y_pred, [-1] + list(KC.int_shape(y_pred)[1:]))
#
#     patches_pred = KC.extract_image_patches(
#         y_pred, kernel, kernel, 'valid', K.image_data_format())
#     patches_true = KC.extract_image_patches(
#         y_true, kernel, kernel, 'valid', K.image_data_format())
#
#     bs, w, h, c1, c2, c3 = KC.int_shape(patches_pred)
#
#     patches_pred = KC.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
#     patches_true = KC.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
#     # Get mean
#     u_true = KC.mean(patches_true, axis=-1)
#     u_pred = KC.mean(patches_pred, axis=-1)
#     # Get variance
#     var_true = K.var(patches_true, axis=-1)
#     var_pred = K.var(patches_pred, axis=-1)
#     # Get covariance
#     covar_true_pred = K.mean(
#         patches_true * patches_pred, axis=-1) - u_true * u_pred
#
#     ssim = (2 * u_true * u_pred + cc1) * (2 * covar_true_pred + cc2)
#     denom = (K.square(u_true) + K.square(u_pred) + cc1) * \
#         (var_pred + var_true + cc2)
#     ssim /= denom
#
#     return K.mean((1.0 - ssim) / 2.0)

def perceptual_loss(y_true, y_pred):
    '''
    Loss function to calculate 2D perceptual loss

    Parameters
    ----------
    y_ture : float
        4D true image numpy array (batches, xres, yres, channels)
    y_pred : float
        4D test image numpy array (batches, xres, yres, channels)

    Returns
    -------
    float
        RMSE between extracted perceptual features

    @author: Akshay Chaudhari <akshay@subtlemedical.com>
    Copyright Subtle Medical (https://www.subtlemedical.com)
    Created on 2018/04/20

    '''

    n_batches, xres, yres, n_channels = K.get_variable_shape(y_true)

    vgg = VGG16(include_top=False,
                weights='imagenet',
                input_shape=(xres, yres, 3))

    loss_model = models.Model(inputs=vgg.input,
                              outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    # Convert to a 3D image and then calculate the RMS Loss
    y_true_rgb = tf.image.grayscale_to_rgb(y_true/K.max(y_true), name=None)
    y_pred_rgb = tf.image.grayscale_to_rgb(y_pred/K.max(y_true), name=None)

    loss = K.mean(K.square(loss_model(y_true_rgb) - loss_model(y_pred_rgb)))

    return loss


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    try:
        # use theano
        return 20.*np.log10(K.max(y_true)) - 10. * np.log10(K.mean(K.square(y_pred - y_true)))
    except:
        denominator = tf_log(tf_constant(10.0))
        return 20.*tf_log(K.max(y_true)) / denominator - 10. * tf_log(K.mean(K.square(y_pred - y_true))) / denominator
    return 0

# segmetnation related loss
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def dice_coef_05(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = tf.cast(K.greater(K.flatten(y_pred),0.5),'float32')
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# segmentation loss
def seg_crossentropy(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)

#added by Jiahong 2019/1/13 to balance the + and - sample sizes
def seg_crossentropy_weighted(y_true, y_pred):
    epsilon = 1e-6
    ratio_one = 0.95
    ratio_zero = 0.05
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    loss = -2*K.mean(ratio_one*y_true_f*tf.log(y_pred_f+epsilon) + ratio_zero*(1-y_true_f)*tf.log(1-y_pred_f+epsilon))
    return loss

def seg_crossentropy_weighted_bycase(y_true, y_pred):
    epsilon = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    pos_sample = K.sum(y_true_f)
    neg_sample = tf.cast(tf.size(y_true_f),tf.float32) - pos_sample
    ratio = neg_sample / pos_sample
    ratio_one = ratio /(ratio + 1)
    ratio_zero = 1 / (ratio + 1)
    loss = -2*K.mean(ratio_one*y_true_f*tf.log(y_pred_f+epsilon) + ratio_zero*(1-y_true_f)*tf.log(1-y_pred_f+epsilon))
    return loss

def roc_auc_score(y_true,y_pred):
    return tflearn.objectives.roc_auc_score(y_pred,y_true)


def precision(y_true, y_pred):
    """Precision metric.
     Only computes a batch-wise average of precision.
     Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    """Recall metric.
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    return 2*((precision(y_true,y_pred)*recall(y_true,y_pred))/(precision(y_true,y_pred)+recall(y_true,y_pred)))

# volume difference, added by Yannan 2018/12/7
def vol_diff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = tf.cast(K.greater(K.flatten(y_pred), 0.5),'float32') # changed from just y_pred
    weight = 1 / (K.sum(y_true_f) + 1)  # weighted edditted by YAnnan 3/18. original value is 1/100000.
    difference = K.abs(K.sum(y_pred_f) - K.sum(y_true_f)) * weight
    return difference/4
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # weight = 1 / 100000
    # difference = K.abs(K.sum(y_pred_f) - K.sum(y_true_f)) * weight
    # return difference
# weighted_dice, weights calculated according to Tmax performance. Added by Yannan
def weighted_dice(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (1.7 * K.sum(y_true_f) + 0.3 * K.sum(y_pred_f) + smooth)
def weighted_dice_loss(y_true, y_pred):
    return 1. - weighted_dice(y_true, y_pred)
def recall_loss(y_true,y_pred):
    return 1. - recall(y_true,y_pred)
# mixed loss


def mixedLoss(weight_l1=0.5, weight_ssim=0.5, weight_perceptual_loss=0):
    if weight_perceptual_loss > 0:
        def loss_func(x, y): return mean_absolute_error(x, y)*weight_l1 + \
            ssim_loss(x, y)*weight_ssim + perceptual_loss(x, y) * \
            weight_perceptual_loss
    else:
        def loss_func(x, y): return mean_absolute_error(
            x, y)*weight_l1 + ssim_loss(x, y)*weight_ssim
    return loss_func

sd_weights = [0.5,0.5,10,1,1/3,0.005]

#def ssim_and_perc_loss(y_true,y_pred):
#    return sd_weights[0]*ssim_loss(y_true,y_pred)+sd_weights[1]*perceptual_loss(y_true,y_pred)


def sce_and_dice_loss(y_true, y_pred):
    return sd_weights[0]*seg_crossentropy(y_true,y_pred)+sd_weights[1]*dice_coef_loss(y_true,y_pred)

def l1_and_ssim_loss(y_true, y_pred):
    return sd_weights[5]*mean_absolute_error(y_true,y_pred)+sd_weights[3]*ssim_loss(y_true,y_pred)

def sce_and_ssim_with_l1_loss(y_true, y_pred):
    return sd_weights[4]*seg_crossentropy(y_true,y_pred)+sd_weights[4]*ssim_loss(y_true,y_pred)+sd_weights[4]*mean_absolute_error(y_true,y_pred)

def sce_dice_and_l2_loss(y_true,y_pred):
    return sd_weights[3]*seg_crossentropy(y_true,y_pred)+sd_weights[3]*dice_coef_loss(y_true,y_pred) \
            + sd_weights[2]*mean_squared_error(y_true,y_pred)

def psnr(im_gt, im_pred):
    return 20*np.log10(np.max(im_gt.flatten())) - 10 * np.log10(np.mean((im_pred.flatten()-im_gt.flatten())**2))

def l1_loss(y_true, y_pred):
    return sd_weights[3]*mean_absolute_error(y_true,y_pred)

def test_loss(y_true,y_pred):
    return sd_weights[0]*dice_coef_loss(y_true,y_pred)+sd_weights[0]*seg_crossentropy_weighted(y_true, y_pred)+sd_weights[3]*mean_absolute_error(y_true,y_pred)+sd_weights[0]*vol_diff(y_true, y_pred)+sd_weights[4]*recall_loss(y_true,y_pred)
# change from unweighted seg seg_crossentropy to weighted. 2019/1/13
def sce_dice_l2_vol_loss(y_true,y_pred):
    # return sd_weights[0]*dice_coef_loss(y_true,y_pred)+sd_weights[0]*seg_crossentropy_weighted(y_true, y_pred)+sd_weights[3]*mean_absolute_error(y_true,y_pred)+sd_weights[0]*vol_diff(y_true, y_pred)
    return dice_coef_loss(y_true,y_pred)+seg_crossentropy_weighted(y_true, y_pred)+mean_absolute_error(y_true,y_pred)+vol_diff(y_true, y_pred)
def weighted_ce_l1_bycase(y_true, y_pred):
    # return seg_crossentropy_weighted_bycase(y_true, y_pred) + mean_absolute_error(y_true,y_pred) + sd_weights[0] * dice_coef_loss(y_true,y_pred) + sd_weights[0]*vol_diff(y_true, y_pred)
    return seg_crossentropy_weighted_bycase(y_true, y_pred) + mean_absolute_error(y_true,y_pred) + 0.5 * dice_coef_loss(y_true,y_pred) + vol_diff(y_true, y_pred) # change from regular dice loss to dice 05
    ## added dice and vol loss on 2/4 by Yannan.

# get error metrics, for psnr, ssimr, rmse, score_ismrm


def getErrorMetrics(im_pred, im_gt, mask=None):

    # flatten array
    im_pred = np.array(im_pred).astype(np.float).flatten()
    im_gt = np.array(im_gt).astype(np.float).flatten()
    if mask is not None:
        mask = np.array(mask).astype(np.float).flatten()
        im_pred = im_pred[mask > 0]
        im_gt = im_gt[mask > 0]
    mask = np.abs(im_gt.flatten()) > 0

    # check dimension
    assert(im_pred.flatten().shape == im_gt.flatten().shape)

    # NRMSE
    rmse_pred = compare_nrmse(im_gt, im_pred)

    # PSNR
    try:
        psnr_pred = compare_psnr(im_gt, im_pred)
    except:
        psnr_pred = psnr(im_gt, im_pred)
        # print('use psnr')

    # ssim
    data_range = np.max(im_gt.flatten()) - np.min(im_gt.flatten())
    ssim_pred = compare_ssim(im_gt, im_pred, data_range=data_range)
    ssim_raw = compare_ssim(im_gt, im_pred)
    score_ismrm = sum((np.abs(im_gt.flatten()-im_pred.flatten())
                       < 0.1)*mask)/(sum(mask)+0.0)*10000

    return {'rmse': rmse_pred, 'psnr': psnr_pred, 'ssim': ssim_pred,
            'ssim_raw': ssim_raw, 'score_ismrm': score_ismrm}
