from collections import namedtuple
import mxnet as mx
import nibabel as nib
import numpy as np
import os

import augment_dataset_helper
import config

Batch = namedtuple('Batch', ['data'])

def read_nii_header(data_path, series_name, suffix, separate_folder=True):
    filename = os.path.join(data_path,series_name,series_name + suffix)
    if not separate_folder:
        filename = os.path.join(data_path,series_name + suffix)
    return nib.load(filename)

def read_image(name, test_data_path, suffixes=config.data_suffixes, separate_folder=True):
    images_list = []
    handle = None

    bboxes = np.zeros(shape=(len(suffixes),) + (2, 3))

    for j, s in enumerate(suffixes):
        image_handle = read_nii_header(test_data_path, name, s, separate_folder)
        handle = image_handle
        image = image_handle.get_data().astype(np.float32)

        mask = image > 0.
        bboxes[j] = augment_dataset_helper.bbox3(mask)
        image = augment_dataset_helper.normalize(image, mask)

        images_list.append(image.reshape((1, 1,) + image.shape))

    bbox_min = np.min(bboxes[:, 0, :], axis=0).ravel().astype(int)
    bbox_max = np.max(bboxes[:, 1, :], axis=0).ravel().astype(int)
    bbox = np.zeros(shape=(2, 3), dtype=np.float)
    bbox[0] = bbox_min
    bbox[1] = bbox_max

    data = np.concatenate(images_list, axis=1)
    data_crop = augment_dataset_helper.resample_np(
        data[:, :, bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]],
        (1, len(suffixes),) + config.brain_reshape_to,
        1)

    return data, data_crop, handle.affine, bbox.reshape((1, 2, 3))

def predict(model, data, data_crop, brain_bbox):

    pred = model.predict(
        Batch([mx.nd.array(data_crop)])
    )
    
    pred_np = pred[0].asnumpy()
    
    pred_flip = model.predict(
        Batch([mx.nd.array(data_crop[:,:,:,::-1,:]),])
    )

    pred_flip_np = pred_flip[0].asnumpy()
    
    pred128 = (pred_np + pred_flip_np[:,:,:,::-1,:])/2.
    pred = pred128

    low = brain_bbox[0,0,:].astype(np.int32)
    high = brain_bbox[0,1,:].astype(np.int32)
    label = np.zeros(shape=(1,pred.shape[1],)+data.shape[2:])

    diff = (brain_bbox[0,1,:]-brain_bbox[0,0,:]).astype(np.int32)

    label[:, :, low[0]:high[0], low[1]:high[1], low[2]:high[2]] = \
        augment_dataset_helper.resample_np(pred,
                 (1,pred.shape[1], diff[0], diff[1], diff[2]),
                 1)

    label = np.argmax(label, axis=1)

    return label



def reject_small_regions(connectivity, ratio=0.25):
    resulting_connectivity = connectivity.copy()
    unique, counts = np.unique(connectivity, return_counts=True)

    all_nonzero_clusters = np.prod(connectivity.shape) - np.max(counts)

    for i in range(unique.shape[0]):
        if counts[i] < ratio * all_nonzero_clusters:
            resulting_connectivity[resulting_connectivity == unique[i]] = 0

    return resulting_connectivity

