import mxnet as mx

import config
import segmentation_model


def one_hot_label(label, classes, name):
    lbl_oh = label.one_hot(classes)
    lbl_oh = lbl_oh.transpose(axes=(0, 5, 2, 3, 4, 1))
    lbl_oh = mx.sym.mean(lbl_oh, axis=5, name=name)

    return lbl_oh

def get_seg_model(data_name, label_name, training):

    data_crop = mx.sym.Variable(data_name)
    label_crop = mx.sym.Variable(label_name)
    lbl_oh = one_hot_label(label_crop,config.seg_output_features, name='onehot_128')

    seg_predict, seg_loss = segmentation_model.get_segmentation_model(data=data_crop, label=lbl_oh, filters_number=config.seg_filters_number,
                                                                          n_outputs=config.seg_output_features, name_prefix='seg_only', training= training)

    symb = mx.sym.BlockGrad(seg_predict)

    if training:
        symb = mx.sym.Group([mx.sym.BlockGrad(seg_predict), seg_loss])

    return symb