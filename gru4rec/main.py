# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
import model
import evaluation

PATH_TO_TRAIN = ''
PATH_TO_TEST = ''

class Args():
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 3
    batch_size = 50
    dropout_p_hidden=1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    
    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})
    args = Args()
    args.n_items = len(data['ItemId'].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        if args.is_training:
            gru.fit(data)
        else:
            res = evaluation.evaluate_sessions_batch(gru, data, valid)
            print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
