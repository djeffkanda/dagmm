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

from utils.utils import check_dir, optimizer_setup
# from TrainTestManager import TrainTestManager, optimizer_setup

from model.AutoEncoder import AutoEncoder as AE
from model.DAGMM import DAGMM
from datamanager.KDDDataset import KDDDataset
from datamanager.DataManager import DataManager
from trainer.AETrainTestManager import AETrainTestManager
from trainer.DAGMMTrainTestManager import DAGMMTrainTestManager
from viz.viz import plot_losses, plot_2D_latent, plot_1D_latent_vs_loss


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 main.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 main.py --model AE [hyper_parameters]'
                                           '\n python3 main.py --model AE --predict',
                                     description="Description...."
                                     )
    parser.add_argument('--model', type=str, default="AE",
                        choices=["AE", "DAGMM", "MLAD"])
    parser.add_argument('--dataset', type=str, default="kdd", choices=["kdd", "hsherbrooke"])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD", "RMSProp"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
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

    # Loading the data
    dataset = KDDDataset(path='../data/kddcup_data_10_percent_corrected.csv')

    normal_data = dataset.get_data_index_by_label(label=0)

    # split data in train and test sets
    train_set, test_set = dataset.split_train_test(.2)
    dm = DataManager(train_set, test_set, batch_size=batch_size)

    # safely create save path
    check_dir(args.save_path)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)
    # print(dataset.get_shape())
    # model = AE(dataset.get_shape()[1], [60, 30, 10, 2], fa='tanh')
    #
    # model_trainer = AETrainTestManager(model=model,
    #                                    dm=dm,
    #                                    loss_fn=nn.MSELoss(reduction='none'),
    #                                    optimizer_factory=optimizer_factory,
    #                                    )
    #
    # metrics = model_trainer.train(200)
    # codes, labels, loss, losses_items = model_trainer.evaluate_on_test_set()
    # # train_loss
    # plot_losses(metrics['train_loss'], metrics['val_loss'])
    # plot_2D_latent(codes, labels)
    # # plot_1D_latent_vs_loss(codes, labels, losses_items)
    #
    # print('Test one')

    # Test DAGMM

    model = DAGMM(dataset.get_shape()[1],
                  [60, 30, 10, 1],
                  fa='tanh',
                  gmm_layers=[10, 4]
                  )

    model_trainer = DAGMMTrainTestManager(model=model,
                                          dm=dm,
                                          loss_fn=nn.MSELoss(reduction='none'),
                                          optimizer_factory=optimizer_factory,
                                          )

    metrics = model_trainer.train(20)
    codes, labels, loss, losses_items = model_trainer.evaluate_on_test_set()
    # train_loss
    plot_losses(metrics['train_loss'], metrics['val_loss'])
    plot_2D_latent(codes, labels)
    # plot_1D_latent_vs_loss(codes, labels, losses_items)

    print('Test one')
