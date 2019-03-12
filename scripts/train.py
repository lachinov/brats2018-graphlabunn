import argparse
import mxnet as mx
import numpy as np
import os
import pickle
import random


import config
import iterators
import joint_model
import metrics_seg
import model as mod

    
def train_seg_model(args, segmentation_model_name,continue_training=False, num_epoch=1):
    """
    Train segmentation model
    :param args: argparse arguments
    :param segmentation_model_name: name of the model
    :param continue_training: load from snapshot
    :param num_epoch: number of epoches
    :return:
    """
    sym, arg_params, aux_params = None, None, None
    model_path = os.path.join(config.models_root_path,segmentation_model_name,segmentation_model_name)

    if continue_training and os.path.isdir(os.path.join(config.models_root_path,segmentation_model_name)):
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, 0)
    
    train_ids = pickle.load(open(os.path.join(args.aug_data_dir,config.train_ids_filename), "rb"))
    val_ids = pickle.load(open(os.path.join(args.aug_data_dir,config.val_ids_filename), "rb"))

    print len(train_ids)

    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    batch_size = config.batch_size

    data_names = ['data_crop']
    label_names = ['label_crop']

    train_iter = iterators.BrainSegmentationIterator(
        data_directory=args.aug_data_dir,
        data_filenames=train_ids,
        data_suffix=config.suffix_data,
        data_cropped_suffix=config.suffix_data_cropped,
        label_suffix=config.suffix_label,
        label_cropped_suffix=config.suffix_label_cropped,
        batch_size=batch_size,
        data_names=data_names,
        label_names=label_names,
        shuffle=True,
        flip=0.5,
        drop_channel=0.1
    )

    valid_iter = iterators.BrainSegmentationIterator(
        data_directory=args.aug_data_dir,
        data_filenames=val_ids,
        data_suffix=config.suffix_data,
        data_cropped_suffix=config.suffix_data_cropped,
        label_suffix=config.suffix_label,
        label_cropped_suffix=config.suffix_label_cropped,
        batch_size=batch_size,
        data_names=data_names,
        label_names=label_names,
        shuffle=False,
        flip=0.,
        drop_channel=0.
    )
    
    
    p_train_iter = mx.io.PrefetchingIter(train_iter)
    p_valid_iter = mx.io.PrefetchingIter(valid_iter)


    symb = joint_model.get_seg_model('data_crop', 'label_crop', True)

    p_train_iter.reset()
    p_valid_iter.reset()


    gpus = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    model = mod.Model(symbol=symb, name=segmentation_model_name, data_names=data_names,
                        label_names=label_names, context=gpus, loss_index=[-1],
                        path=config.models_root_path, rewrite_dir=True
                        )

    lr_sch = mx.lr_scheduler.FactorScheduler(step=len(train_ids), factor=args.lr_decay)
    model.train(p_train_iter, p_valid_iter,
                initializer=mx.init.Xavier(),
                optimizer='sgd',
                optimizer_params={
                    'learning_rate': args.lr,
                    'lr_scheduler': lr_sch,
                    'momentum': args.momentum,
                    'wd':args.wd
                },
                arg_params=arg_params,
                aux_params=aux_params,
                train_metrics=[
                    metrics_seg.Dice(classes=4, label_index=0, pred_index=0),
                    metrics_seg.DisplayLoss(name='seg_loss', loss_index=-1),
                ],
                val_metrics=[
                    metrics_seg.DisplayLoss(name='seg_loss', loss_index=-1),
                    metrics_seg.Dice(classes=4, label_index=0, pred_index=0),
                ],
                num_epoch=num_epoch,
                track_metric = 'Dice',
                comparator = lambda x, y : x[0] > y[0],
                default_val = np.array([0,0,0,0]),
                epoch_end_callback=[mx.callback.do_checkpoint(prefix='backup', period=20)],
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train cascaded UNet")
    parser.add_argument('--name', type=str, default=config.segmentation_model_name, help='model\'s name')
    parser.add_argument('--data_dir', type=str, default=config.data_directory, help='the input data directory')
    parser.add_argument('--aug_data_dir', type=str, default=config.augumented_directory, help='the augmented data directory')
    parser.add_argument('--gpus', type=str, default='0,1', help='the gpus will be used, e.g "0,1,2,3"')
    #training options
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=config.seg_lr_decay_factor, help='lr decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--epoches', type=int, default=config.seg_train_epoches, help='number of training epoches')
    parser.add_argument('--continue_training', type=bool, default=True, help='continue training or launch from scratch')

    args = parser.parse_args()

    train_seg_model(args, args.name,args.continue_training,args.epoches)