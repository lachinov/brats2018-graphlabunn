import nibabel as nib
import numpy as np
import os
import pickle
from scipy.ndimage import interpolation
import SimpleITK as sitk
import random
from tqdm import tqdm

import config

def read_nii(data_path, series_name, suffix):
    """
    read nii file
    :param data_path: path to the dataset folder
    :param series_name: name of the series
    :param suffix: modality suffix
    :return: numpy array
    """
    filename = os.path.join(data_path,series_name, series_name + suffix)
    image = nib.load(filename)
    return np.array(image.get_data())


def read_nii_header(data_path, series_name, suffix):
    """
    read nii header
    :param data_path: path to the dataset folder
    :param series_name: name of the series
    :param suffix: modality suffix
    :return: nibabel header
    """
    filename = os.path.join(data_path,series_name,series_name + suffix)
    return nib.load(filename)

def bbox3(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2,3) and contents [min,max]
    """
    rows = np.any(img, axis=1)
    rows = np.any(rows, axis=1)

    cols = np.any(img, axis=0)
    cols = np.any(cols, axis=1)

    slices = np.any(img, axis=0)
    slices = np.any(slices, axis=0)

    rows = np.where(rows)
    cols = np.where(cols)
    slices = np.where(slices)
    if (rows[0].shape[0] > 0):
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]
        smin, smax = slices[0][[0, -1]]

        return np.array([[rmin, cmin, smin], [rmax, cmax, smax]])
    return np.array([[-1,-1,-1],[0,0,0]])


def resample(image, transform, interpolator = sitk.sitkLinear):
    """
    Resample image according to transformation
    :param image: input itk image
    :param transform: itk transformation
    :param interpolator: itk interpolator
    :return: resampled itk image
    """
    reference_image = image
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def resample_np(data, output_shape, order):
    """
    Resample input np array to match specified shape
    :param data: input nd array
    :param output_shape: desired shape
    :param order: interpolation spline order
    :return: interpolated image
    """
    assert(len(data.shape) == len(output_shape))
    factor = [float(o) / i for i,o in zip(data.shape, output_shape)]
    return interpolation.zoom(data, zoom=factor, order = order)

def augment_data(array, bspline, interpolation):
    """
    augment input nd array with itk bspline
    :param array: nd array
    :param bspline: itk transform
    :param interpolation: itk interpolation
    :return: transformed nd array
    """
    image = sitk.GetImageFromArray(array)

    image_resampled = sitk.GetArrayFromImage(resample(image, bspline, interpolation))
    return image_resampled.astype(np.float32)

def gen_bspline(data_image):
    """
    generate bspline interpolation
    :param data_image: input image
    :return: bspline tranformation
    """
    grid_size = 6
    image = sitk.GetImageFromArray(data_image)
    bspline = sitk.BSplineTransformInitializer(image, [grid_size/2, grid_size, grid_size])
    scale = 12 + random.random()*8
    p = np.random.random((grid_size+3, grid_size+3, grid_size+3, 3))*scale - scale/2

    bspline.SetParameters(p.ravel())
    return bspline

def normalize(image, mask):
    """
    perform data normalization
    :param image: input nd array
    :param mask: corresponding foreground mask
    :return: normalized array
    """
    ret = image.copy()
    image_masked = np.ma.masked_array(ret, ~(mask))
    ret[mask] = ret[mask] - np.mean(image_masked)
    ret[mask] = ret[mask] / np.var(image_masked) ** 0.5
    ret[ret > 5.] = 5.
    ret[ret < -5.] = -5.
    ret += 5.
    ret /= 10

    ret[~mask] = 0.

    return ret

def augment_subdataset_split(args, subfolder, id_list, data_suffixes, aug_name, augment=True, multiplier=1):
    """
    augment specific dataset split
    :param args: command line arguments with data_dir and aug_data_dir
    :param subfolder: sub-dataset folder. Either hgg or lgg
    :param id_list: list of sequences
    :param data_suffixes: list of modality suffixes
    :param aug_name: name of output files
    :param augment: perform augmentation of not
    :param multiplier: dataset multiplier
    :return: list of augmented ids, corresponding brain bboxes and tumor bboxes
    """
    input_shape = read_nii_header(os.path.join(args.data_dir,subfolder), id_list[0], data_suffixes[0]).get_data().shape
    brain_bboxes = np.zeros(shape=(len(id_list) * multiplier, 2, 3), dtype=int)
    tumor_bboxes = np.zeros(shape=(len(id_list) * multiplier, 2, 3), dtype=int)

    for m in tqdm(range(len(id_list) * multiplier)):
        i = m % len(id_list)
        f = id_list[i]

        data_size = config.input_modalities
        sample = np.zeros(shape=(1, data_size,) + input_shape, dtype=np.float32)

        bspline = gen_bspline(np.zeros(shape=input_shape))

        bboxes = np.zeros(shape=(len(data_suffixes),) + (2, 3))

        for j, s in enumerate(data_suffixes):
            image_handle = read_nii_header(os.path.join(args.data_dir,subfolder), f, s)
            image = image_handle.get_data().astype(np.float32)

            if augment:
                image = augment_data(image, bspline, sitk.sitkLinear)

            bboxes[j] = bbox3(image)
            mask = image > 0
            image = normalize(image, mask)

            sample[0, j] = image

        bbox_min = np.min(bboxes[:, 0, :], axis=0).ravel().astype(int)
        bbox_max = np.max(bboxes[:, 1, :], axis=0).ravel().astype(int)
        bbox = np.zeros(shape=(2, 3), dtype=int)
        bbox[0] = bbox_min
        bbox[1] = bbox_max
        brain_bboxes[m] = bbox
        sample_cropped = resample_np(sample[:,:,bbox_min[0]:bbox_max[0],bbox_min[1]:bbox_max[1],bbox_min[2]:bbox_max[2]],
                                    (1, data_size,)+config.brain_reshape_to,
                                    1)
        

        np.save(os.path.join(args.aug_data_dir,aug_name + str(m) + config.suffix_data), sample.astype(np.float32), allow_pickle=False)
        np.save(os.path.join(args.aug_data_dir,aug_name + str(m) + config.suffix_data_cropped), sample_cropped.astype(np.float32), allow_pickle=False)

        label_handle = read_nii_header(os.path.join(args.data_dir,subfolder), f, config.suffix_seg)
        label_data = label_handle.get_data().astype(np.float32)
        for key, value in config.dataset_transform_dict.iteritems():
            label_data[label_data == key] = value

        if augment:
            label_data = augment_data(label_data, bspline, sitk.sitkNearestNeighbor)

        tumor_bboxes[m] = bbox3(label_data > 0)
        tumor_bboxes[m,0] = tumor_bboxes[m,0] - bbox[0]
        tumor_bboxes[m,1] = tumor_bboxes[m,1] - bbox[0]

        label_data = label_data.reshape((1, 1,) + label_data.shape)
        label_data_cropped = resample_np(label_data[:,:,bbox_min[0]:bbox_max[0],bbox_min[1]:bbox_max[1],bbox_min[2]:bbox_max[2]],
                                    (1, 1,)+config.brain_reshape_to,
                                    0)
        
        np.save(os.path.join(args.aug_data_dir,aug_name + str(m) + config.suffix_label), label_data.astype(np.float32), allow_pickle=False)
        np.save(os.path.join(args.aug_data_dir,aug_name + str(m) + config.suffix_label_cropped), label_data_cropped.astype(np.float32), allow_pickle=False)

    return [aug_name + str(i) for i in range(len(id_list) * multiplier)], brain_bboxes, tumor_bboxes


def augment_subdataset(args, train_ids, val_ids, gluoma_type):
    """
    augment subdataset. Either hgg or lgg
    :param args: command line arguments with data_dir and aug_data_dir
    :param train_ids: list of training sequences
    :param val_ids: list of validation sequences
    :param gluoma_type: HGG or LGG
    :return: list of train ids, list of val ids, train brain bboxes, val brain bboxes, train tumor bboxes, val tumor bboxes
    """
    #augment
    new_train_ids, new_train_brain_bbox, new_train_tumor_bbox = augment_subdataset_split(args=args, subfolder=gluoma_type, id_list=train_ids, aug_name=gluoma_type + '_aug',
                                                                                         augment=True, multiplier=config.dataset_multiplier-1, data_suffixes=config.data_suffixes)
    #copy original data
    old_train_ids, old_train_brain_bbox, old_train_tumor_bbox = augment_subdataset_split(args=args, subfolder=gluoma_type, id_list=train_ids, aug_name=gluoma_type + '_orig',
                                                                                         augment=False, multiplier=1, data_suffixes=config.data_suffixes)
    #copy validation data
    old_val_ids, old_val_brain_bbox, old_val_tumor_bbox = augment_subdataset_split(args=args, subfolder=gluoma_type, id_list=val_ids, aug_name=gluoma_type + '_val',
                                                                                   augment=False, multiplier=1, data_suffixes=config.data_suffixes)

    return new_train_ids + old_train_ids, old_val_ids, np.concatenate([new_train_brain_bbox, old_train_brain_bbox],axis=0), old_val_brain_bbox,\
           np.concatenate([new_train_tumor_bbox, old_train_tumor_bbox],axis=0), old_val_tumor_bbox

def augment_dataset(args, train_ids_hgg, val_ids_hgg,train_ids_lgg, val_ids_lgg):
    """
    augment whole dataset both hgg and lgg
    this function writes trains_ids, train_brain_bbox, train_tumor_bbox, val_ids, val_brain_bbox, val_tumor_bbox to the numpy dataset folder
    :param args: command line arguments with data_dir and aug_data_dir
    :param train_ids_hgg: list of hgg training sequences
    :param val_ids_hgg: list of hgg validation sequences
    :param train_ids_lgg: list of lgg training sequences
    :param val_ids_lgg: list of lgg validation sequences
    :return:
    """
    train_ids_hgg, val_ids_hgg, train_brain_bbox_hgg, val_brain_bbox_hgg, train_tumor_bbox_hgg, val_tumor_bbox_hgg =\
        augment_subdataset(args, train_ids_hgg, val_ids_hgg, 'HGG')
    train_ids_lgg, val_ids_lgg, train_brain_bbox_lgg, val_brain_bbox_lgg, train_tumor_bbox_lgg, val_tumor_bbox_lgg =\
        augment_subdataset(args, train_ids_lgg, val_ids_lgg, 'LGG')

    train_ids = train_ids_hgg + train_ids_lgg
    val_ids = val_ids_hgg + val_ids_lgg
    train_brain_bbox = np.concatenate([train_brain_bbox_hgg, train_brain_bbox_lgg],axis=0)
    val_brain_bbox = np.concatenate([val_brain_bbox_hgg , val_brain_bbox_lgg  ], axis=0)
    train_tumor_bbox = np.concatenate([train_tumor_bbox_hgg, train_tumor_bbox_lgg],axis=0)
    val_tumor_bbox = np.concatenate([val_tumor_bbox_hgg , val_tumor_bbox_lgg  ], axis=0)

    assert (len(train_ids) == train_brain_bbox.shape[0])
    assert (len(val_ids) == val_brain_bbox.shape[0])
    assert (len(train_ids) == train_tumor_bbox.shape[0])
    assert (len(val_ids) == val_tumor_bbox.shape[0])

    pickle.dump(train_ids, open(os.path.join(args.aug_data_dir,config.train_ids_filename), "wb"))
    pickle.dump(train_brain_bbox, open(os.path.join(args.aug_data_dir,config.train_brain_bbox_filename), "wb"))
    pickle.dump(train_tumor_bbox, open(os.path.join(args.aug_data_dir, config.train_tumor_bbox_filename), "wb"))
    pickle.dump(val_ids, open(os.path.join(args.aug_data_dir, config.val_ids_filename), "wb"))
    pickle.dump(val_brain_bbox, open(os.path.join(args.aug_data_dir, config.val_brain_bbox_filename), "wb"))
    pickle.dump(val_tumor_bbox, open(os.path.join(args.aug_data_dir, config.val_tumor_bbox_filename), "wb"))