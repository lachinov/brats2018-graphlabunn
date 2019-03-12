import gc
import mxnet as mx
from mxboard import SummaryWriter
import numpy as np
import os
import pickle
import shutil
import time

import utils

class Model:
    """
    Model - class encapsulating training and predicting functionality
    It if capable of performing training, selecting best model according given metric,
    writing logs for tensorboard, relaunching training from the last snapshot,
    saving and loading model and its state
    """
    def __init__(self, symbol, name, **kwargs):
        """
        :param symbol: model symbol
        :param name: model name
        :param kwargs: training related arguments like
        :param path: path to the models root directory (default: ./)
        :param loss_index: loss index in the output (default: -1)
        :param data_names: name of the data inputs
        :param label_names: names of the label inputs
        :param context: execution context
        :param group2ctxs: group of contexts (optional)
        :param fixed_param_names: names of the fixed parameters (optional)
        :param arg_params: model states (optional)
        :param aux_params: model states (optional)
        :param rewrite_dir: rewrite model directory if it exists
        """

        self.name = name
        self.path = kwargs.get('path', './')

        self.path = self.path + name + '/'
        self.logs_path = self.path + '/logs/'
        self.sw = SummaryWriter(logdir=self.logs_path, flush_secs=5)
        #training information
        self.symbol = symbol
        self.module = None
        self.epoch = 0
        self.train_metric = dict()
        self.val_metric = dict()
        #arguments
        self.init_args = kwargs
        self.loss_index = kwargs.get('loss_index', -1)
        module_keys = ['data_names', 'label_names', 'context', 'group2ctxs', 'fixed_param_names']
        self.module_desc = dict((k, kwargs[k]) for k in module_keys if k in kwargs)

        self.arg_params = kwargs.get('arg_params', None)
        self.aux_params = kwargs.get('aux_params', None)

        if 'rewrite_dir' in kwargs:
            if os.path.exists(self.path) and kwargs['rewrite_dir']:
                shutil.rmtree(self.path)
            os.mkdir(self.path)
            os.mkdir(self.logs_path)

    def train(self, train_iterator, val_iterator, **kwargs):
        """
        Train model

        :param train_iterator: training iterator
        :param val_iterator: validation iterator
        :param kwargs: training related arguments like
        :param arg_params: model states (optional)
        :param aux_params: model states (optional)
        :param initializer: model weights initializer
        :param optimizer: optimization algorithm
        :param optimizer_params: parameters of the optimization algorithm
        :param train_metrics: list of training metrics
        :param val_metrics: list of validation metrics
        :param num_epoch: number of epochs to train
        :param default_val: default value of validation metric being tracked
        :param track_metric: name of the metric being tracked
        :param comparator: metric comparator
        :param epoch_end_callback: list of epoch end callbacks
        """
        if 'arg_params' in kwargs:
            self.arg_params = kwargs['arg_params']

        if 'aux_params' in kwargs:
            self.aux_params = kwargs['aux_params']
        
        #create new module
        if self.module is None:
            self.module = mx.mod.Module(symbol=self.symbol, **self.module_desc)

        #initialize
        self.module.bind(data_shapes=train_iterator.provide_data, label_shapes=train_iterator.provide_label, for_training=True)
        self.module.init_params(initializer=kwargs['initializer'])
        if self.arg_params is not None:
            self.module.set_params(arg_params=self.arg_params, aux_params=self.aux_params, allow_missing=True, allow_extra=True)
        self.module.init_optimizer(optimizer=kwargs['optimizer'], optimizer_params=kwargs['optimizer_params'])

        print 'model has been binded'
        #prepare dicts for metrics
        self.train_metric = dict()
        self.val_metric = dict()

        for m in kwargs['train_metrics']:
            self.train_metric[m.name] = []

        for m in kwargs['val_metrics']:
            self.val_metric[m.name] = []

        start_epoch = self.epoch
        num_epoch = kwargs['num_epoch']

        self.sw.add_graph(self.module.symbol)

        #number of processed batches
        global_step = 0
        #initialize best validation score
        self.best_val = kwargs['default_val']
        
        self.save()

        for i in range(start_epoch, num_epoch):
            train_iterator.reset()
            val_iterator.reset()
            tic = time.time()
            
            global_step = self._train_one_epoch(train_iterator, kwargs['train_metrics'],
                                           self.train_metric, i, global_step)

            self.arg_params, self.aux_params = self.module.get_params()
            self.epoch = i
            
            self._evaluate_and_save(val_iterator, kwargs['val_metrics'], kwargs['track_metric'],
                                    self.val_metric, i, kwargs['comparator'])

            mx.model._multiple_callbacks(kwargs['epoch_end_callback'], i, self.module.symbol, self.arg_params, self.aux_params)

            tac = time.time()

            print 'Epoch %d, time %s\n' % (i, tac - tic)

    def predict(self, batch):
        """
        Perform prediction
        :param batch: input data batch
        :return: model's output
        """
        if self.module is None:
            desc = self.module_desc
            desc['label_names'] = None
            #print desc

            self.symbol = self.symbol.get_internals()[self.symbol.list_outputs()[0]]

            self.module = mx.mod.Module(symbol=self.symbol, **desc)
            #bind data shape

            #print self.module._label_shapes, data_shapes
            data_shapes = [(name,batch.data[i].shape) for i, name in enumerate(desc['data_names'])]
            #label_shapes = [(name, batch.label[i].shape) for i, name in enumerate(desc['label_names'])]

            self.module.bind(for_training=False, data_shapes=data_shapes)
            self.module.set_params(arg_params=self.arg_params, aux_params=self.aux_params, allow_missing=True)
        self.module.forward(batch)
        return self.module.get_outputs()
    
    def clear(self):
        #clear module and release gpu memory
        self.module = None
        gc.collect()
        return

    def _train_one_epoch(self, train_iter, train_metrics, train_metrics_results, epoch, global_step):

        for m in train_metrics:
            m.reset()

        for batch in train_iter:

            self.module.forward_backward(batch)  # compute predictions
            self.module.update()

            for m in train_metrics:
                self.module.update_metric(m, batch.label)  # accumulate prediction accuracy

            outputs = self.module.get_outputs()
            for i in self.loss_index:
                outputs[i].wait_to_read()
                loss = np.mean(outputs[i].asnumpy())
                utils.log_var(loss, 'loss' + str(i), global_step, self.sw)

            global_step += 1

        for m in train_metrics:
            train_metrics_results[m.name].append(m.get()[1])
            utils.log_var(m.get()[1], 'train_' + m.name, epoch, self.sw)
            print('Epoch %d, Training %s %s' % (epoch, m.name, m.get()[1]))
        return global_step

    def _evaluate_and_save(self, eval_iter, val_metrics, track_metric, val_metrics_results, epoch, comparator):
        for m in val_metrics:
            m.reset()

        for batch in eval_iter:
            self.module.forward(batch, is_train=False)  # compute predictions

            for m in val_metrics:
                m.update(batch.label, self.module.get_outputs())

        val = 0.0
        for m in val_metrics:
            if m.name == track_metric:
                val = m.get()[1]

            utils.log_var(m.get()[1], 'val_' + m.name, epoch, self.sw)
            val_metrics_results[m.name].append(m.get()[1])
            print('Epoch %d, Validation %s %s' % (epoch, m.name, m.get()[1]))

        if comparator(val, self.best_val):
            self.best_val = val
            self.save()
            print 'model saved'

    def save(self):
        #save training state
        self.module.save_checkpoint(self.path+self.name, 0)
        #save training metadata
        pickle.dump(self.train_metric, open(os.path.join(self.path,'train_metric.p'), 'wb'))
        pickle.dump(self.val_metric, open(os.path.join(self.path,'val_metric.p'), 'wb'))
        pickle.dump(self.epoch, open(os.path.join(self.path,'epoch.p'), 'wb'))
        pickle.dump(self.module_desc, open(os.path.join(self.path,'module_desc.p'), 'wb'))

    def load(self, path, load_symbol=False):
        #load training state
        self.path = path
        model_prefix = os.path.join(path,self.name)
        model_number = 0
        sym, self.arg_params, self.aux_params = mx.model.load_checkpoint(model_prefix, model_number)
        if load_symbol:
            self.symbol = sym
            
        #load training metadata 
        self.train_metric = pickle.load(open(os.path.join(path,'train_metric.p'), 'rb'))
        self.val_metric = pickle.load(open(os.path.join(path, 'val_metric.p'), 'rb'))
        self.epoch = pickle.load(open(os.path.join(path, 'epoch.p'), 'rb'))
        self.module_desc = pickle.load(open(os.path.join(path, 'module_desc.p'), 'rb'))

        return

    def load_params(self, sym, arg_params, aux_params):
        if sym is not None:
            self.symbol = sym
        self.arg_params = arg_params
        self.aux_params = aux_params