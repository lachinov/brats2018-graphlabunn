import mxnet as mx
import numpy as np
import os


def transform_to_center_size_format(bboxes):
    bboxes_sizes = bboxes[:, 1, :] - bboxes[:, 0, :]
    bboxes_center = (bboxes[:, 1, :] + bboxes[:, 0, :]) / 2.0

    bboxes[:, 0, :] = bboxes_center
    bboxes[:, 1, :] = bboxes_sizes
    return bboxes

def read_numpy(data_path, series_name, suffix):
    filename = os.path.join(data_path,series_name+suffix)
    return np.load(filename)

def log_var(var, tag, global_step, sw):
    if isinstance(var, np.ndarray):
        var = var.ravel()
        
        for i in range(var.shape[0]):
            sw.add_scalar(tag=tag+str(i), value=var[i], global_step=global_step)
    else:
        sw.add_scalar(tag=tag, value=var, global_step=global_step)

def calculate_metric(metrics, labels, preds):
    for m in metrics:
        m.update(labels, preds)
        
    return metrics

def print_summary(symbol, shape=None, label_shape=None):
    output_shape = None
    if label_shape is not None and shape is not None :
        arg_shape, output_shape, aux_shape = symbol.infer_shape(data=shape, label=label_shape)
    elif label_shape is None and shape is not None :
        arg_shape, output_shape, aux_shape = symbol.infer_shape(data=shape)
    elif label_shape is not None and shape is None :
        arg_shape, output_shape, aux_shape = symbol.infer_shape(label=label_shape)
    print output_shape

def print_shape(net, input_shape):
    arg_shape, output_shape, aux_shape = net.infer_shape(data=input_shape)
    print output_shape
def get_shape(net, input_shape):
    arg_shape, output_shape, aux_shape = net.infer_shape(data=input_shape)
    return output_shape