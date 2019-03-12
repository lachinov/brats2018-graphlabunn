# dataset paths
# path to the root directory. HGG and LGG should be child directories of data_directory
data_directory = '../dataset/'

# directory to store augmentation dataset. should be empty in order to run augmentation
augumented_directory = '../dataset/numpy_dataset/augmented/128/'

# directory to models root
models_root_path = '../models/'
# segmentation model name. directory models_root_path/segmentation_model_name will be created
segmentation_model_name = 'only_seg'

# augmentator
train_ids_filename = 'train_ids.p'
train_brain_bbox_filename = 'train_brain_bbox.p'
train_tumor_bbox_filename = 'train_tumor_bbox.p'
val_ids_filename = 'val_ids.p'
val_brain_bbox_filename = 'val_brain_bbox.p'
val_tumor_bbox_filename = 'val_tumor_bbox.p'
# number of augmentation circles
# 1 - copy original data
# 2 - original data + augmented dataset
# 3 - original data + two augmented datasets
# 4 - ...
dataset_multiplier = 2

# suffixes for numpy representation
suffix_data = "_data.npy"
suffix_data_cropped = "_data_cropped.npy"
suffix_label = "_label.npy"
suffix_label_cropped = "_label_cropped.npy"

# suffixes for original interpretation in dataset
suffix_t1 = "_t1.nii.gz"
suffix_t2 = "_t2.nii.gz"
suffix_flair = "_flair.nii.gz"
suffix_t1ce = "_t1ce.nii.gz"
suffix_seg = "_seg.nii.gz"
# file suffixes to form a data tensor
data_suffixes = [suffix_t1, suffix_t2, suffix_flair, suffix_t1ce]
# labels mappings
# transform label 4 to 3
dataset_transform_dict = {
    4 : 3,
}

# sampling settings
# resample brain region to this size
brain_reshape_to = (128, 128, 128)
# resample tumor region to this size
tumor_reshape_to = (96, 96, 96)

# network settings
# batch size. should be compatible with number of gpus
batch_size = 4
input_modalities = len(data_suffixes)
slope = 1e-2
# network activation 'relu','lrelu','prelu','elu','softplus'
activation = 'relu'
# number of encoders
encoder_groups = 4
# due to use of grouped convs number of feature channels in each encoder is N / encoder_groups
# this parameter multiplies number of feature channels for every encoder making it
# encoder_channel_multiplier * N / encoder_groups
encoder_channel_multiplier = 2


# segmentation variables
# number of output classes
seg_output_features = 4
# base number of feature channels in segmentation network
seg_filters_number = 4
# mxnet conv workspace
seg_workspace = 1024
# lr decay factor for every epoch
seg_lr_decay_factor = 0.99
# total number of epoches
seg_train_epoches = 500
