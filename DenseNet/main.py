from __future__ import print_function

import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run CIFAR10 experiment')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--nb_epoch', default=30, type=int, help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7, help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1, help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16, help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12, help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4, help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False, help='Save a plot of the network architecture')

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    from DenseNet import nn

    nn.run(args.batch_size, args.nb_epoch, args.depth, args.nb_dense_block, args.nb_filter, args.growth_rate,
           args.dropout_rate,
           args.learning_rate,
           args.weight_decay,
           args.plot_architecture)
