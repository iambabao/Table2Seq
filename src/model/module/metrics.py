# -*- coding: utf-8 -*-

"""
@Author     : Bao
@Date       : 2020/2/20 21:39
@Desc       :
"""

import tensorflow as tf


def get_loss(labels, logits):
    mask = tf.cast(tf.math.not_equal(labels, 0), tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))  # loss by token

    return loss


def get_accuracy(labels, logits):
    mask = tf.cast(tf.math.not_equal(labels, 0), tf.float32)
    pred_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    accuracy = tf.cast(tf.equal(labels, pred_ids), tf.float32)
    accuracy = tf.reduce_mean(tf.reduce_sum(mask * accuracy, axis=-1) / tf.reduce_sum(mask, axis=-1))

    return accuracy
