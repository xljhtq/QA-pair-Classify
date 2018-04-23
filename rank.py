#!/usr/bin/env python
# encoding=utf-8

import tensorflow as tf


class Ranking(object):
    def __init__(self, max_len_left, max_len_right,
                 vocab_size, embedding_size, filter_sizes,
                 num_filters, num_hidden,
                 fix_word_vec, word_vocab,
                 l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, max_len_left], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, max_len_right], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        print(self.input_left)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if fix_word_vec:
                word_vec_trainable = False
                wordInitial = tf.constant(word_vocab.word_vecs)
                W = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                                    initializer=wordInitial,
                                    dtype=tf.float32)
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_Embedding")

            # [batch_size, max_length, embedding_size, 1]
            # 官方定义为：[batch, in_height, in_width, in_channels]
            self.embedded_chars_left = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_left), -1)
            self.embedded_chars_right = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_right), -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_left = []
        pooled_outputs_right = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name="W")  # W: [filter_height, filter_width, in_channels, out_channels], 与input对应
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(self.embedded_chars_left, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # conv: [batch_size, 20-2+1, 1, out_channels]

                pooled = tf.nn.max_pool(h,
                                        ksize=[1, max_len_left - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")  # pooled: [batch_size, 1, 1, out_channels]
                pooled_outputs_left.append(pooled)
            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_right, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, max_len_right - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs_right.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_left = tf.reshape(tf.concat(axis=3, values=pooled_outputs_left), [-1, num_filters_total],
                                      name='h_pool_left')
        self.h_pool_right = tf.reshape(tf.concat(axis=3, values=pooled_outputs_right), [-1, num_filters_total],
                                       name='h_pool_right')

        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable("W",
                                shape=[num_filters_total, num_filters_total],
                                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.h_pool_left, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.h_pool_right), 1, keep_dims=True)

        # Make input for classification
        self.new_input = tf.concat(axis=1, values=[self.h_pool_left, self.sims, self.h_pool_right], name='new_input')

        # hidden layer
        l2_loss = tf.constant(0.0)
        with tf.name_scope("hidden"):
            W = tf.get_variable("W_hidden",
                                shape=[2 * num_filters_total + 1, num_hidden],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W_output",
                                shape=[num_hidden, 2],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.prob = tf.nn.softmax(self.scores, name='prob')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print("self.prob", self.prob.name)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            print("self.loss", self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            print("self.accuracy", self.accuracy.name)
