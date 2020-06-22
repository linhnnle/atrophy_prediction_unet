import matplotlib

matplotlib.use('Agg')

''' basic dependencies '''
from time import time

from subtle_fileio import *
from network import *
from subtle_metrics import *
from subtle_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf


def get_volume_info_from_dir(dict_sample_info, list_subjects,ext_data,num_of_aug):
    '''
    list_subjects: e.g ['300001','300002']
    dict_sample_info: '/data/stroke/'
    '''
    result_list = []
    num_subject = len(list_subjects)
    print('############################ total subject number:', num_subject)
    for subj in list_subjects:
        dir_list_subj = dict_sample_info + subj
        result_list = result_list + get_volume_info_from_subjects(dir_list_subj,ext_data,num_of_aug)

    print('get {0} subject with {1} {2} file pairs'.format(
            num_subject, len(result_list), ext_data))
    print('example:', result_list[0])
    return result_list



def get_volume_info_from_file_pair(input_filename_list, output_filename_list, subject_dir,
                                   data_ext):
    input_len = len(input_filename_list)
    output_len = len(output_filename_list)

    result_list = []

    print('samples:', subject_dir, input_filename_list, output_filename_list)
    _, shape, _ = get_data_from_ext(os.path.join(subject_dir, input_filename_list), data_ext, return_mean=False)
    mean = [-1, -1, -1]
    result_list.append({'filepath_inputs': [os.path.join(subject_dir, input_filename_list)],
                        'filepath_output': os.path.join(subject_dir, output_filename_list),
                        'data_shape': shape,
                        'values_mean': mean,
                        'sample_weight': 1,
                        'indexes_slice_excluded': []})
    return result_list


'''this function finds datafolder that with target files'''
def get_volume_info_from_subjects(subject_list,ext_data,num_of_aug):
    # subject_num = len(subject_list)
    result_list = []
    for index in range(0,num_of_aug):## modified by Yannan 2018/11/27
        result_list = result_list + get_volume_info_from_file_pair('inputs_aug{0}.hdf5'.format(index), 'output_aug{0}.hdf5'.format(index), subject_list,ext_data)
    return result_list


def get_train_sample_list(volume_info_list, slice_expand_num,axis_slice):
    sample_file_num = len(volume_info_list)
    list_samples = []
    for index in range(sample_file_num):
        # get sample info
        info = volume_info_list[index]
        inputs = info['filepath_inputs']
        output = info['filepath_output']
        # dim
        slice_num = info['data_shape'][axis_slice]
        weight = info['sample_weight']

        list_samples += [[inputs, output, x, weight] for x in range(slice_expand_num, slice_num - slice_expand_num)]
    return list_samples


def get_train_and_validation_samples(training_samples, split):
    # validation_split = 500
    sample_num = len(training_samples)
    np.random.seed(0)
    training_samples = [training_samples[x] for x in np.random.permutation(sample_num)]
    if split > 1:
        validation_samples = training_samples[-split:]  # .tolist()
        training_samples = training_samples[:int(sample_num - split)]  # .tolist()
    else:
        validation_samples = training_samples[-int(sample_num * split):]  # .tolist()
        training_samples = training_samples[:int(sample_num * (1 - split))]  # .tolist()
    print('train on {0} samples and validation on {1} samples'.format(
        len(training_samples), len(validation_samples)))
    return training_samples, validation_samples


def stroke_train_img_segmentation(dir_of_train = '/data/yuanxie/stroke_preprocess173/', subj_list = ['01002','30082A'],val_list = '', extension_data='hdf5',
                 log_dir = '../log',
                 dir_ckpt = '/Users/admin/stroke_DL/ckpt_stroke',
                 filename_checkpoint = 'model_stroke_test.ckpt',
                 filename_model = 'model_stroke_test.json',
                 model_select = 'dropout', num_epochs=10,
                 shape_px_width = 128, shape_px_height =128, num_contrast_input = 7, num_channel_output = 1,
                 output_range_low = 0., out_put_range_high = 1.0, with_batch_norm = True, batch_size = 91,
                 lr_init = 0.005, num_conv_per_pooling = 4, num_poolings = 3, num_workers = 16, num_slice_expand = 2,
                 validation_split = 0.1, keras_memory=0.85,gpu='2',num_of_aug=1, loss_mode = 'regular'):
    '''
    setup gpu
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    '''
    exclusion slices
    '''
    dict_slices_excluded = {}

    '''
    get dataset info
    '''

    '''
    pre setup parameters related to sampling
    '''

    axis_slice = 0

    max_queue_size = num_workers * 2

    output_range = None
    # dimension
    num_slice_25d = num_slice_expand * 2 + 1
    index_slice_mid = num_slice_expand
    num_channel_input = num_slice_25d * num_contrast_input

    shape_resize = [shape_px_width, shape_px_height]
    img_rows = shape_resize[0]
    img_cols = shape_resize[1]
    # generator settings
    generator_resize_data = False
    generator_normalize_data = False
    generator_mask_data = False
    generator_sanitize_data = False
    generator_axis_slice = 0
    print('setup parameters')

    '''
    setup model
    '''
    setKerasMemory(keras_memory)
    loss_monitoring = [dice_coef, seg_crossentropy, precision, recall, l1_loss]


    if loss_mode == 'bycase':
        loss_func = weighted_ce_l1_bycase
        print('using weighted bce by case + l1 loss')
    else:
        loss_func = sce_dice_l2_vol_loss
        print('using dice+sce+l1+volume as loss')


    '''
    model parameters
    '''
    model_parameters = {
        "num_channel_input": num_channel_input,
        "num_channel_output": num_channel_output,
        "img_rows": img_rows,
        "img_cols": img_cols,
        "output_range": np.array(output_range),
        "loss_function": loss_func,
        "metrics_monitor": loss_monitoring,
        "num_poolings": num_poolings,
        "num_conv_per_pooling": num_conv_per_pooling,
        "with_bn": with_batch_norm,
        "with_baseline_concat": True,
        "with_baseline_addition": -1,
        "activation_conv": 'relu',  # 'selu','relu','elu'
        "activation_output": 'sigmoid',
        "kernel_initializer": 'he_normal',
        "verbose": 1
    }

    if model_select == 'dropout':
        model = dropoutResUNet(**model_parameters)
    model.summary()

    '''
    get files from each dataset
    '''

    list_volume_info = get_volume_info_from_dir(dir_of_train, subj_list, extension_data,num_of_aug)

    list_volume_info_val = get_volume_info_from_dir(dir_of_train, val_list, extension_data,1)
    num_sample_file = len(list_volume_info)

    '''
    define samples
    '''
    list_samples_train = get_train_sample_list(list_volume_info, num_slice_expand,axis_slice)
    list_samples_val = get_train_sample_list(list_volume_info_val, num_slice_expand,axis_slice)
    num_sample = len(list_samples_train)

    '''
    setup model
    '''
    # dir_ckpt = '/Users/admin/stroke_DL/ckpt_stroke'

    filepath_checkpoint = os.path.join(dir_ckpt, filename_checkpoint)
    filepath_model = os.path.join(dir_ckpt, filename_model)
    print('setup parameters')

    '''
    init model
    '''
    # setup lr
    model.optimizer = Adam(lr=lr_init)
    print('train model:', filename_checkpoint)
    print('parameter count:', model.count_params())

    print('train from scratch')
    # setup callbacks
    # changed by Larry Liu from filename_checkpoint to filepath_checkpoint 09/22
    model_checkpoint = ModelCheckpoint(filepath_checkpoint,
                                       monitor='val_loss',
                                       save_best_only=True,verbose=1)
    model_tensorboard = TensorBoard(log_dir="{0}/{1}_{2}".format(log_dir,
                                                                 time(), filename_checkpoint.split('.')[0]))
    model_callbacks = [model_checkpoint, model_tensorboard]

    '''
    save model to file
    '''
    model_json = model.to_json()
    with open(filepath_model, "w") as json_file:
        json_file.write(model_json)

    '''
    log for model setup
    '''
    print('train model description:', filepath_model)
    print('train model weights checkpoint:', filepath_checkpoint)
    print('parameter count:', model.count_params())

    '''
    define generator
    '''
    # details inside generator
    params_generator = {'dim_x': img_rows,
                        'dim_y': img_cols,
                        'dim_z': num_channel_input,
                        'dim_25d': num_slice_25d,
                        'dim_output': 1,
                        'batch_size': batch_size,
                        'shuffle': True,
                        'verbose': 1,
                        'scale_data': 1,
                        'scale_baseline': 0.0,
                        'normalize_per_sample': False,
                        'para': {
                            # 'index_mid':index_slice_mid,
                            # 'clip_output':[0,20],
                            # 'augmentation':[2,2,2,1],
                            'num_contrast_input': num_contrast_input
                        },
                        'normalize_data': generator_normalize_data,
                        'resize_data': generator_resize_data,
                        'mask_data': generator_mask_data,
                        'sanitize_data': generator_sanitize_data,
                        'axis_slice': generator_axis_slice
                        }
    print('generator parameters:', params_generator)

    '''
    init generator
    '''
    training_generator = DataGenerator(**params_generator).generate(list_volume_info, list_samples_train)
    validation_generator = DataGenerator(**params_generator).generate(list_volume_info_val, list_samples_val)

    '''
    train model with generator
    change num_epochs as needed
    '''

    steps_per_epoch = int(len(list_samples_train) / batch_size)
    validation_steps = int(len(list_samples_val) / batch_size)
    print('train batch size:{0}, step per epoch:{1}, step per val epoch:{2}'.format(
        batch_size, steps_per_epoch, validation_steps))

    history = model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        callbacks=model_callbacks,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        max_queue_size=max_queue_size,
        workers=num_workers,
        use_multiprocessing=True ###
    )
    print('model successfully trained. please find model and checkpoint in', filepath_checkpoint)

    model_json = model.to_json()
    with open(filepath_model, "w") as json_file:
        json_file.write(model_json)
    model.save(filepath_model[:-5]+'.h5')

    del model
