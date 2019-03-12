import mxnet as mx
import numpy as np
import random
import utils

class BrainSegmentationIterator(mx.io.DataIter):
    """
    Data loading iterator class, performes data loading and augmentation

    :param data_directory: path to the converted dataset
    :param data_filenames: list of filenames to iterate over
    :param data_suffix: suffix of data file
    :param data_cropped_suffix: suffix of the data file with cropped brain region
    :param label_suffix: suffix of label file
    :param data_cropped_suffix: suffix of the label file with cropped brain region
    :param batch_size: size of the batch
    :param data_names: list with input data names. Currenlty data_names[0] is in use
    :param label_names: list with input label names. Currenlty label_names[0] is in use
    :param shuffle: whether to shuffle the inputs every epoch or not
    :param flip: probabilty to flip the image along X and Y axises during augmentation
    :param drop_channel: probabilty to fill individual input channel with gaussian noise
    """
    def __init__(self, data_directory, data_filenames, data_suffix, data_cropped_suffix, label_suffix, label_cropped_suffix,
                 batch_size, data_names, label_names, shuffle, flip=0.5, drop_channel=0.1):
        super(BrainSegmentationIterator, self).__init__()
        self.data_directory = data_directory
        self.data_filenames = data_filenames
        self.data_suffix = data_suffix
        self.data_cropped_suffix = data_cropped_suffix
        self.label_suffix = label_suffix
        self.label_cropped_suffix = label_cropped_suffix
        self.flip = flip
        self.drop_channel = drop_channel

        self.batch_size = batch_size
        self.shuffle = shuffle

        data_shape = utils.read_numpy(data_directory, data_filenames[0], data_suffix).shape
        crop_shape = utils.read_numpy(data_directory, data_filenames[0], data_cropped_suffix).shape

        self.data_shape = (batch_size,) + data_shape[1:]
        self.data_crop_shape = (batch_size,) + crop_shape[1:]
        self.label_shape = (batch_size,1) + data_shape[2:]
        self.label_crop_shape = (batch_size,1,) + crop_shape[2:]

        self._provide_data = [mx.io.DataDesc(name=data_names[0], shape=self.data_crop_shape, layout='NCDHW')]
        self._provide_label = [mx.io.DataDesc(name=label_names[0], shape=self.label_crop_shape, layout='NCDHW')]

        self.current_batch = 0

        self.index = np.arange(len(data_filenames), dtype=np.int)

        self.data_crop_out = np.zeros(shape=self.data_crop_shape, dtype=float)
        self.label_crop_out = np.zeros(shape=self.label_crop_shape, dtype=float)

        print self.data_crop_shape
        print self.data_shape
        
    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            random.shuffle(self.index)
        self.current_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def is_batch_valid(self, batch_no):
        if (batch_no + 1) * self.batch_size - 1 >= len(self.data_filenames):
            return False
        return True

    def load_batch(self, batch_no):
        indices = self.index[batch_no * self.batch_size: (batch_no + 1) * self.batch_size]
        filenames = [self.data_filenames[i] for i in indices]
        
        for i, (f, idx) in enumerate(zip(filenames, indices)):
            self.data_crop_out[i] = utils.read_numpy(self.data_directory, f, self.data_cropped_suffix)

            for c in range(self.data_shape[1]):
                if random.random() < self.drop_channel:
                    self.data_crop_out[i,c] = np.random.randn(*self.data_crop_out[i,c].shape)

            self.label_crop_out[i] = utils.read_numpy(self.data_directory, f, self.label_cropped_suffix)

            if random.random() < self.flip:
                self.data_crop_out[i] = self.data_crop_out[i,:,::-1,:,:]
                self.label_crop_out[i] = self.label_crop_out[i,:,::-1,:,:]
                
            if random.random() < self.flip:
                self.data_crop_out[i] = self.data_crop_out[i,:,:,::-1,:]
                self.label_crop_out[i] = self.label_crop_out[i,:,:,::-1,:]

    def next(self):

        if not self.is_batch_valid(self.current_batch):
            raise StopIteration

        self.load_batch(self.current_batch)

        data_crop_out = mx.nd.array(self.data_crop_out)   
        label_crop_out = mx.nd.array(self.label_crop_out)

        self.current_batch += 1

        ret_label = [label_crop_out]
        ret_data = [data_crop_out]
        return mx.io.DataBatch(ret_data, ret_label)