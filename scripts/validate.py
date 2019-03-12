import argparse
import mxnet as mx
import numpy as np
import os
import pickle
import predict_utils

import config
import metrics_seg
import model
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate the model")
    parser.add_argument('--name', type=str, default=config.segmentation_model_name, help='model\'s name')
    args = parser.parse_args()

    val_ids = pickle.load(open(os.path.join(config.augumented_directory,config.val_ids_filename), "rb"))
    val_brain_bboxes = pickle.load(open(os.path.join(config.augumented_directory,config.val_brain_bbox_filename), "rb")).astype(
        np.float)
    val_tumor_bboxes = pickle.load(open(os.path.join(config.augumented_directory,config.val_tumor_bbox_filename), "rb")).astype(
        np.float)

    data_names = ['data_crop']
    label_names = None

    m = model.Model(None, args.name, data_names=data_names,
                    label_names=label_names, context=[mx.gpu(0)], loss_index=[-1])

    m.load(os.path.join(config.models_root_path,args.name), True)
    m.module_desc['context'] = [mx.gpu(0)]

    scores = {}
    s = 0
    for i, v in enumerate(val_ids):
        label = utils.read_numpy(config.augumented_directory, v, config.suffix_label)
        data = utils.read_numpy(config.augumented_directory, v, config.suffix_data)
        data_crop = utils.read_numpy(config.augumented_directory, v, config.suffix_data_cropped)
        brain_bbox = val_brain_bboxes[i,None]

        dice = metrics_seg.Dice(4)

        pred_label = predict_utils.predict(m, data, data_crop, brain_bbox)

        dice.update([mx.nd.array(label)], [mx.nd.array(pred_label.reshape(label.shape))])

        s += dice.get()[1]

        scores[v] = dice.get()[1]
        print v


    print 'mean Dice scores {}'.format(s / len(val_ids))
    scores_list = [v for k, v in scores.iteritems()]
    print 'median Dice scores {}'.format(np.median(scores_list, axis=0))

    print '\n\n==============================================='
    print 'Samplewise scores\n'
    for k, v in scores.iteritems():
        print k, v