import argparse
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split

import augment_dataset_helper
import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply augmentation for training dataset")
    parser.add_argument('--data_dir', type=str, default=config.data_directory, help='the input data directory')
    parser.add_argument('--aug_data_dir', type=str, default=config.augumented_directory,
                        help='the augmented data directory')

    args = parser.parse_args()

    train_ids_hgg = next(os.walk(os.path.join(args.data_dir, 'HGG')))[1]
    train_ids_lgg = next(os.walk(os.path.join(args.data_dir, 'LGG')))[1]
    seed = 100
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(args.aug_data_dir):
        os.makedirs(args.aug_data_dir)
    elif len(os.listdir(args.aug_data_dir)) == 0:
        print 'Warning: Reusing existing empty directory'
    else:
        print 'Error: Dataset directory already exists'
        exit(-1)

    train_ids_hgg, val_ids_hgg = train_test_split(train_ids_hgg, test_size=0.25, random_state=seed)
    train_ids_lgg, val_ids_lgg = train_test_split(train_ids_lgg, test_size=0.25, random_state=seed)

    print 'hgg val ids:'
    print val_ids_hgg
    print '\nlgg val ids:'
    print val_ids_lgg

    augment_dataset_helper.augment_dataset(args, train_ids_hgg, val_ids_hgg, train_ids_lgg, val_ids_lgg)
