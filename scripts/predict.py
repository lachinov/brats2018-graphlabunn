import argparse
import mxnet as mx
import nibabel as nib
import numpy as np
import os
from skimage import morphology

import config
import model
import predict_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make predictions for series in the test folder and store results in the output folder")
    parser.add_argument('--name', type=str, default=config.segmentation_model_name, help='model\'s name')
    parser.add_argument('--test_data_dir', type=str, default='../dataset/val/', help='testing data directory')
    parser.add_argument('--out_data_dir', type=str, default='../dataset/submit/', help='submission directory')
    parser.add_argument('--is_single_series', dest='is_single_series', action='store_true')
    parser.set_defaults(is_single_series=False)

    args = parser.parse_args()

    test_ids = next(os.walk(args.test_data_dir))[1]
    print test_ids

    data_names = ['data_crop']
    label_names = None

    m = model.Model(None, args.name, data_names=data_names,
                    label_names=label_names, context=[mx.gpu(0)], loss_index=[-1])

    m.load(os.path.join(config.models_root_path,args.name), True)
    m.module_desc['context'] = [mx.gpu(0)]

    suffixes=config.data_suffixes
    if args.is_single_series:
        test_ids = ['']
        suffixes = ['t1.nii.gz', 't2.nii.gz', 't1ce.nii.gz', 'flair.nii.gz']

    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)

    for idx in test_ids:
        data, data_crop, affine, brain_bbox = predict_utils.read_image(idx, suffixes=suffixes,
                                                                       test_data_path=args.test_data_dir,
                                                                       separate_folder=(args.is_single_series == False))

        label = predict_utils.predict(m, data, data_crop, brain_bbox)

        label = label.reshape(data.shape[2:]).astype(np.float32)

        for key, value in config.dataset_transform_dict.iteritems():
            label[label == value] = key

        connected_regions = morphology.label(label > 0)

        clusters = predict_utils.reject_small_regions(connected_regions, 0.1)
        label[clusters == 0] = 0

        pred_nii = nib.Nifti1Image(label, affine)

        print str(idx) + ' ' + str(np.mean(label > 0)) + ' ' + str(np.unique(label))

        if args.is_single_series:
            nib.save(pred_nii, os.path.join(args.out_data_dir, 'tumor_graphlabunn_class.nii.gz'))
        else:
            nib.save(pred_nii, os.path.join(args.out_data_dir,idx + '.nii.gz'))