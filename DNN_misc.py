import glob
import os
from pathlib import Path

import numpy as np

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as k

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, ReLU

from Disco_tf import distance_corr

def manageOutputdir(outputdir):
    # check if dir exists
    if(os.path.exists(outputdir)):
        i=2
        while(os.path.exists(outputdir+"__"+str(i))):
            i+=1
        outputdir += "__"+str(i)
    Path(outputdir).mkdir(parents=True)
    Path(outputdir+"/plots/").mkdir(parents=True)
    Path(outputdir+"/data/").mkdir(parents=True)
    Path(outputdir+"/network/").mkdir(parents=True)

    return outputdir


# custom metrics
def acc_all(labels, predictions):
    m = tf.keras.metrics.Accuracy()
    weights = tf.reshape(labels[:,2],[tf.size(labels[:,2]).numpy(), 1])
    m.update_state(labels[:,0], predictions, sample_weight=weights)
    return m.result().numpy()

# partial accuracys for weighted
def acc_sig(labels, predictions):
    msk = k.equal(labels[:,0], 1)
    bools = k.equal(tf.boolean_mask(labels[:,0], msk), k.round(tf.boolean_mask(predictions, msk)))
    bools = tf.cast(bools, dtype=tf.float32)
    bools = tf.math.multiply(bools, tf.boolean_mask(labels[:,2], msk))
    return k.mean(bools)
def acc_bkg(labels, predictions):
    msk = k.equal(labels[:,0], 0)
    bools = k.equal(tf.boolean_mask(labels[:,0], msk), k.round(tf.boolean_mask(predictions, msk)))
    bools = tf.cast(bools, dtype=tf.float32)
    bools = tf.math.multiply(bools, tf.boolean_mask(labels[:,2], msk))
    return k.mean(bools)

def nominalLoss_wrapper():
    def nominalLoss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()
        weights = tf.reshape(y_true[:,2],[tf.size(y_true[:,2]).numpy(), 1])
        loss_bce = bce(y_true[:,0], y_pred, sample_weight=weights)
        return loss_bce
    return nominalLoss

def DisCoLoss_wrapper(param):
    def DisCoLoss(y_true, y_pred):
        totalWeight = k.sum(y_true[:,2])
        weights = y_true[:,2]*tf.size(y_true[:,2]).numpy()/totalWeight
        loss_DisCo = param*distance_corr(y_true[:,1], y_pred[:,0], weights)
        return loss_DisCo
    return DisCoLoss

# loss for DisCo
def cross_entropy_DisCo(param):
    def loss(y_true, y_pred):
        # calculating weighted BCE
        bce = tf.keras.losses.BinaryCrossentropy()
        weights = tf.reshape(y_true[:,2],[tf.size(y_true[:,2]).numpy(), 1])
        loss_bce = bce(y_true[:,0], y_pred, sample_weight=weights)

        # calculating DisCo
        # ensuring weights sum up to sample count
        totalWeight = k.sum(y_true[:,2])
        weights = y_true[:,2]*tf.size(y_true[:,2]).numpy()/totalWeight
        loss_DisCo = param * distance_corr(y_true[:,1], y_pred[:,0], weights)

        return loss_bce + loss_DisCo
    return loss

def mean_squared_error_DisCo(param):
    def loss(y_true, y_pred):
        # calculating weighted BCE
        mse = tf.keras.losses.MeanSquaredError()
        weights = tf.reshape(y_true[:,2],[-1, 1])
        loss_mse = mse(y_true[:,0], y_pred, sample_weight=weights)
        print(loss_mse)

        # calculating DisCo
        # ensuring weights sum up to sample count
        totalWeight = k.sum(y_true[:,2])
        weights = y_true[:,2]*tf.size(y_true[:,2]).numpy()/totalWeight
        loss_DisCo = param * distance_corr(y_true[:,1], y_pred[:,0], weights)

        return loss_mse + loss_DisCo
    return loss

def layerblock(L, nodes, kernel_init, dropout_val=0, batchNorm=False):
    if(batchNorm): L = BatchNormalization()(L)

    L = Dense(nodes, kernel_initializer=kernel_init, activation='tanh')(L)
    if(dropout_val > 0): L = Dropout(dropout_val)(L)
    L = ReLU()(L)
    return L
