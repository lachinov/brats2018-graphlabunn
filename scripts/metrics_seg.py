from mxnet import metric
import numpy as np


class DisplayLoss(metric.EvalMetric):
    def __init__(self, eps=1e-12, name='loss', loss_index=1,
                 output_names=None, label_names=None):
        super(DisplayLoss, self).__init__(
            name, eps=eps,
            output_names=output_names, label_names=label_names)
        self.eps = eps
        self.loss_index = loss_index

    def update(self, labels, preds):
        loss = preds[self.loss_index]
        self.sum_metric += loss.sum().asnumpy()
        self.num_inst += loss.shape[0]

class Dice(metric.EvalMetric):
    """
    Dice metric, accumulates Dice metric in a vector. every vector element corresponds to separate class, except
    the background class. Instead of background foreground Dice is calculated
    """
    def __init__(self, classes, eps=1e-12, name='Dice', label_index=0, pred_index=0,
                 output_names=None, label_names=None):
        super(Dice, self).__init__(
            name, eps=eps,
            output_names=output_names, label_names=label_names)
        self.eps = eps
        self.classes = classes
        self.label_index =label_index
        self.pred_index = pred_index
        self.image_count = 0
        self.intersection = np.zeros(shape=(self.classes,))
        self.union = np.zeros(shape=(self.classes,))

    def update(self, labels, preds):
        label = labels[self.label_index].asnumpy()
        pred = preds[self.pred_index].asnumpy()
            
        if pred.shape[1] > 1:
            pred = np.argmax(pred, axis=1)

        assert np.prod(label.shape) == np.prod(pred.shape)

        result = np.zeros(shape=(self.classes,))

        for i in range(label.shape[0]):
            for c in range(1, self.classes):
                label_ = (label[i] == c)
                pred_ = (pred[i] == c)

                pred_no = pred_.sum()
                label_no = label_.sum()
                intersect_no = np.logical_and(pred_, label_).sum()
                union_no = pred_no + label_no
                if intersect_no > 0:
                    result[c] += 2.0 * intersect_no / float(union_no)

        for i in range(label.shape[0]):
            label_ = label[i] > 0
            pred_ = pred[i] > 0

            pred_no = pred_.sum()
            label_no = label_.sum()
            intersect_no = (pred_ * label_).sum()
            union_no = pred_no + label_no
            if intersect_no > 0:
                result[0] += 2.0 * intersect_no / float(union_no)

        self.sum_metric += result
        self.num_inst += label.shape[0]