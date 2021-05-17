#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
PhD Project
Authors: D'Jeff Kanda
"""

import argparse
import torch.optim as optim
import torch.nn as nn
import copy

from utils import get_data, check_dir
from viz import plot_query_strategy_metrics, plot_compare_to_random_metrics, plot_all_metrics
from query_strats.DataManager import DataManager as DM
from TrainTestManager import TrainTestManager, optimizer_setup

from models.SimpleModel import SimpleModel
from models.SENet import SENet
from models.ResNet import ResNet
from models.DenseNet import DenseNet
from query_strats.RandomQueryStrategy import RandomQueryStrategy
from query_strats.EntropyQueryStrategy import EntropyQueryStrategy
from query_strats.LCQueryStrategy import LCQueryStrategy
from query_strats.MSQueryStrategy import MSQueryStrategy


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model SENet [hyper_parameters]'
                                           '\n python3 train.py --model SENet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Active learning is used to prioritise sample "
                                                 "selection for in the training process"
                                     )
    parser.add_argument('--model', type=str, default="BasicCNN",
                        choices=["BasicCNN", "SENet", "ResNet", "DenseNet"])
    parser.add_argument('--dataset', type=str, default="mnistfashion", choices=["cifar100", "mnistfashion"])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--initial_data_ratio', type=float, default=0.01,
                        help='Percentage of training data randomly selected on first iteration of active'
                             'learning process')
    parser.add_argument('--query_strategy', type=str, default='Random',
                        choices=['Random', 'LC', 'Margin', 'Entropy'],
                        help='Type of strategy to use for querying data in active learning process')
    parser.add_argument('--query_size', type=int, default=100,
                        help='Size of sample to label per query')
    parser.add_argument('--train_set_threshold', type=float, default=1,
                        help='Percentage of training data as threshold to stop active learning process')
    parser.add_argument('--data_aug', action='store_true',
                        help="Data augmentation")
    parser.add_argument('--mode', type=str, default='Single', choices=['Single', 'Compare', 'All'])
    parser.add_argument('--save_path', type=str, default="./", help='The path where the output will be stored,'
                                                                    'model weights as well as the figures of '
                                                                    'experiments')
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    initial_data_ratio = args.initial_data_ratio
    train_set_threshold = args.train_set_threshold
    query_size = args.query_size
    data_augment = args.data_aug

    # Loading the data
    train_set, test_set = get_data(data_augment, args.dataset)
    if val_set is not None:
        dm = DM(train_set, test_set, batch_size=batch_size, validation=val_set,
                initial_train_dataset_ratio=initial_data_ratio)
    else:
        dm = DM(train_set, test_set, batch_size=batch_size,
                initial_train_dataset_ratio=initial_data_ratio)
    # safely create save path
    check_dir(args.save_path)

    # adjust num_query with threshold
    num_query = len(train_set) * train_set_threshold * (1 - initial_data_ratio) // query_size

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.dataset == 'mnistfashion':
        num_channels = 1
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_channels = 3
        num_classes = 100

    if args.model == 'BasicCNN':
        model = SimpleModel(num_channels=num_channels, num_classes=num_classes)
    elif args.model == 'SENet':
        model = SENet(num_channels=num_channels, num_classes=num_classes)
    elif args.model == 'ResNet':
        model = ResNet(num_channels=num_channels, num_classes=num_classes)
    elif args.model == 'DenseNet':
        model = DenseNet(num_channels=num_channels, num_classes=num_classes)

    if args.mode == 'Single' or args.mode == 'Compare':
        if args.query_strategy == 'Random':
            query_strategy = RandomQueryStrategy(dm)
        elif args.query_strategy == 'LC':
            query_strategy = LCQueryStrategy(dm)
            pass
        elif args.query_strategy == 'Margin':
            query_strategy = MSQueryStrategy(dm)
            pass
        elif args.query_strategy == 'Entropy':
            query_strategy = EntropyQueryStrategy(dm)

        model_trainer = TrainTestManager(model=model,
                                         querier=query_strategy,
                                         loss_fn=nn.CrossEntropyLoss(),
                                         optimizer_factory=optimizer_factory,
                                         )

        if args.mode == 'Single':
            print('Training using {} Query Strategy'.format(query_strategy.__class__.__name__))
            model_trainer.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size)
            plot_query_strategy_metrics(model_trainer, args.save_path)
        elif args.mode == 'Compare':
            random_query = RandomQueryStrategy(copy.deepcopy(dm))
            random_model_trainer = TrainTestManager(model=model,
                                                    querier=random_query,
                                                    loss_fn=nn.CrossEntropyLoss(),
                                                    optimizer_factory=optimizer_factory)
            # Training
            print('Training using Random Query Strategy')
            random_model_trainer.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size)

            print('Training using {} Query Strategy'.format(query_strategy.__class__.__name__))
            model_trainer.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size)

            plot_compare_to_random_metrics(model_trainer, random_model_trainer, args.save_path)

    elif args.mode == 'All':
        random = RandomQueryStrategy(copy.deepcopy(dm))
        entropy = EntropyQueryStrategy(copy.deepcopy(dm))
        lc = LCQueryStrategy(copy.deepcopy(dm))
        margin = MSQueryStrategy(copy.deepcopy(dm))
        del dm

        random_manager = TrainTestManager(model=copy.deepcopy(model),
                                          querier=random,
                                          loss_fn=nn.CrossEntropyLoss(),
                                          optimizer_factory=copy.deepcopy(optimizer_factory))
        print('Training using Random Query Strategy')
        random_manager.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size,
                             save_path=args.save_path)

        entropy_manager = TrainTestManager(model=model,
                                           querier=entropy,
                                           loss_fn=nn.CrossEntropyLoss(),
                                           optimizer_factory=optimizer_factory)
        print('Training using Entropy Query Strategy')
        entropy_manager.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size,
                              save_path=args.save_path)

        lc_manager = TrainTestManager(model=model,
                                      querier=lc,
                                      loss_fn=nn.CrossEntropyLoss(),
                                      optimizer_factory=optimizer_factory)
        print('Training using Least Confidence Query Strategy')
        lc_manager.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size,
                         save_path=args.save_path)

        margin_manager = TrainTestManager(model=model,
                                          querier=margin,
                                          loss_fn=nn.CrossEntropyLoss(),
                                          optimizer_factory=optimizer_factory)
        print('Training using Margin Query Strategy')
        margin_manager.train(num_epochs=num_epochs, num_query=num_query, query_size=query_size,
                             save_path=args.save_path
                             )

        plot_all_metrics(random_manager, entropy_manager, lc_manager, margin_manager, args.save_path)
