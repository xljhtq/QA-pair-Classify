#!/usr/bin/env python
# encoding=utf-8

import tensorflow as tf


class Ranking_DSSMCNN(object):
    def leaky_relu(self, x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

    def general_loss(self, logits, labels):
        scores = tf.matmul(logits, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))
        labels2 = tf.matmul(labels, tf.constant([0, 1], shape=[2, 1], dtype=tf.float32))

        tmp = tf.multiply(tf.subtract(1.0, labels2), tf.square(scores))
        temp2 = tf.multiply(labels2, tf.square(tf.maximum(tf.subtract(1.0, scores), 0.0)))
        sum = tmp + temp2
        return sum

    def batch_normalization(self, conv, num_filters, need_test=True):
        beta = tf.Variable(tf.ones(num_filters), name='beta')
        offset = tf.Variable(tf.zeros(num_filters), name='offset')
        bnepsilon = 1e-5

        axis = list(range(len(conv.shape) - 1))
        batch_mean, batch_var = tf.nn.moments(conv, axis)
        ema = tf.train.ExponentialMovingAverage(0.99)
        update_ema = ema.apply([batch_mean, batch_var])

        def mean_var_with_update(batch_mean, batch_var):
            with tf.control_dependencies([update_ema]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # if need_test == True:
        #     mean, var = mean_var_with_update(ema.average(batch_mean), ema.average(batch_var))
        # else:
        #     mean, var = batch_mean, batch_var
        mean, var = tf.cond(tf.equal(need_test, True),
                            lambda: mean_var_with_update(ema.average(batch_mean), ema.average(batch_var)),
                            lambda: (batch_mean, batch_var))

        Ybn = tf.nn.batch_normalization(conv, mean, var, offset, beta, bnepsilon)

        return Ybn, mean, var, beta, offset

    def batch_same(self, conv, mean, var, beta, offset):
        bnepsilon = 1e-5
        Ybn = tf.nn.batch_normalization(conv, mean, var, offset, beta, bnepsilon)
        return Ybn

    def __init__(self, max_len,
                 vocab_size, embedding_size, filter_sizes,
                 num_filters, num_hidden,
                 fix_word_vec, word_vocab,
                 C=4,
                 l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, max_len], name="input_left")
        self.input_centre = tf.placeholder(tf.int32, [None, max_len], name="input_centre")
        self.input_right = tf.placeholder(tf.int32, [None, max_len], name="input_right")
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
                print("fix_word_vec")
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_Embedding")

            # [batch_size, max_length, embedding_size, 1]
            self.embedded_chars_left = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_left), -1)
            self.embedded_chars_centre = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_centre), -1)
            self.embedded_chars_right = tf.expand_dims(tf.nn.embedding_lookup(W, self.input_right), -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_left = []
        pooled_outputs_right = []
        pooled_outputs_centre = []

        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                            name="W_filter-%s" % filter_size)  # W: [filter_height, filter_width, in_channels, out_channels], 与input对应
            with tf.name_scope("conv-maxpool-left-%s" % filter_size):
                conv = tf.nn.conv2d(self.embedded_chars_left, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")  # conv: [batch_size, 25-2+1, 1, out_channels]
                conv_BN, mean, var, beta, offset = self.batch_normalization(conv, num_filters)
                h = tf.nn.tanh(conv_BN)

                (a, b, c, d) = h.get_shape()
                h_i_List = []
                for i in range(C):
                    if i == C - 1:
                        h_i_List.append(
                            tf.slice(h, [0, i * int(b / C), 0, 0], [int(a), int(b - i * int(b / C)), int(c), int(d)]))
                        break
                    h_i_List.append(tf.slice(h, [0, i * int(b / C), 0, 0], [int(a), int(b / C), int(c), int(d)]))

                h_outputs = []
                for h0 in h_i_List:
                    if h0 == h_i_List[-1]:
                        pooled = tf.nn.max_pool(h0,
                                                ksize=[1, b - i * int(b / C), 1, 1],
                                                strides=[1, 1, 1, 1],
                                                padding='VALID',
                                                name="pool")
                        h_outputs.append(pooled)
                        break
                    pooled = tf.nn.max_pool(h0,
                                            ksize=[1, int(b / C), 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name="pool")  # pooled: [batch_size, 1, 1, out_channels]
                    h_outputs.append(pooled)

                self.h_stack_left = h_stack_left = tf.stack(values=h_outputs, axis=4)
                (a, b, c, d, e) = h_stack_left.shape
                self.h_reshape_left = h_reshape_left = tf.reshape(h_stack_left, [-1, 1, 1, int(d * e)])
                pooled_outputs_left.append(h_reshape_left)

            with tf.name_scope("conv-maxpool-right-%s" % filter_size):
                conv = tf.nn.conv2d(self.embedded_chars_right, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv_BN = self.batch_same(conv, mean, var, beta, offset)
                h = tf.nn.tanh(conv_BN)

                (a, b, c, d) = h.shape
                h_i_List = []
                for i in range(C):
                    if i == C - 1:
                        h_i_List.append(
                            tf.slice(h, [0, i * int(b / C), 0, 0], [int(a), int(b - i * int(b / C)), int(c), int(d)]))
                        break
                    h_i_List.append(tf.slice(h, [0, i * int(b / C), 0, 0], [int(a), int(b / C), int(c), int(d)]))

                h_outputs = []
                for h0 in h_i_List:
                    if h0 == h_i_List[-1]:
                        pooled = tf.nn.max_pool(h0,
                                                ksize=[1, b - i * int(b / C), 1, 1],
                                                strides=[1, 1, 1, 1],
                                                padding='VALID',
                                                name="pool")
                        h_outputs.append(pooled)
                        break
                    pooled = tf.nn.max_pool(h0,
                                            ksize=[1, int(b / C), 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name="pool")  # pooled: [batch_size, 1, 1, out_channels]
                    h_outputs.append(pooled)

                h_stack_right = tf.stack(values=h_outputs, axis=4)
                (a, b, c, d, e) = h_stack_right.shape
                h_reshape_right = tf.reshape(h_stack_right, [-1, 1, 1, int(d * e)])
                pooled_outputs_right.append(h_reshape_right)

            with tf.name_scope("conv-maxpool-centre-%s" % filter_size):
                conv = tf.nn.conv2d(self.embedded_chars_centre, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv_BN = self.batch_same(conv, mean, var, beta, offset)
                h = tf.nn.tanh(conv_BN)

                (a, b, c, d) = h.shape
                h_i_List = []
                for i in range(C):
                    if i == C - 1:
                        h_i_List.append(
                            tf.slice(h, [0, i * int(b / C), 0, 0], [int(a), int(b - i * int(b / C)), int(c), int(d)]))
                        break
                    h_i_List.append(tf.slice(h, [0, i * int(b / C), 0, 0], [int(a), int(b / C), int(c), int(d)]))

                h_outputs = []
                for h0 in h_i_List:
                    if h0 == h_i_List[-1]:
                        pooled = tf.nn.max_pool(h0,
                                                ksize=[1, b - i * int(b / C), 1, 1],
                                                strides=[1, 1, 1, 1],
                                                padding='VALID',
                                                name="pool")
                        h_outputs.append(pooled)
                        break
                    pooled = tf.nn.max_pool(h0,
                                            ksize=[1, int(b / C), 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name="pool")  # pooled: [batch_size, 1, 1, out_channels]
                    h_outputs.append(pooled)

                h_stack_centre = tf.stack(values=h_outputs, axis=4)
                (a, b, c, d, e) = h_stack_centre.shape
                h_reshape_centre = tf.reshape(h_stack_centre, [-1, 1, 1, int(d * e)])
                pooled_outputs_centre.append(h_reshape_centre)

        ##### Combine all the pooled features [batch_size,num_filters_total]
        print(pooled_outputs_left)
        print(pooled_outputs_centre)
        num_filters_total = C * num_filters * len(filter_sizes)
        self.h_pool_left = tf.reshape(tf.concat(axis=3, values=pooled_outputs_left), [-1, num_filters_total],
                                      name='h_pool_left')
        self.h_pool_centre = tf.reshape(tf.concat(axis=3, values=pooled_outputs_centre), [-1, num_filters_total],
                                        name='h_pool_centre')
        self.h_pool_right = tf.reshape(tf.concat(axis=3, values=pooled_outputs_right), [-1, num_filters_total],
                                       name='h_pool_right')

        print(self.h_pool_left)
        print(self.h_pool_centre)
        l2_loss = tf.constant(0.0)
        W = tf.get_variable("W_hidden",
                            shape=[num_filters_total, num_hidden],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        print("----------------------------------------------------------------------------")
        ######1. FC layer & dropout  ## tanh(), leaky relu()
        with tf.name_scope("hidden_dropout"):
            with tf.name_scope("hidden_left"):
                xw_plus_b = tf.nn.xw_plus_b(self.h_pool_left, W, b, name="hidden_output_left")
                xw_plus_b_BN, mean, var, beta, offset = self.batch_normalization(xw_plus_b, num_hidden)
                self.hidden_output_left = tf.nn.tanh(xw_plus_b_BN)
                print(mean)
            with tf.name_scope("dropout_left"):
                self.h_drop_left = tf.nn.dropout(self.hidden_output_left, self.dropout_keep_prob,
                                                 name="hidden_output_drop_left")

            ###2. FC layer & dropout
            with tf.name_scope("hidden_centre"):
                xw_plus_b = tf.nn.xw_plus_b(self.h_pool_centre, W, b, name="hidden_output_centre")
                xw_plus_b_BN = self.batch_same(xw_plus_b, mean, var, beta, offset)
                self.hidden_output_centre = tf.nn.tanh(xw_plus_b_BN)
                print(mean)
            with tf.name_scope("dropout_centre"):
                self.h_drop_centre = tf.nn.dropout(self.hidden_output_centre, self.dropout_keep_prob,
                                                   name="hidden_output_drop_centre")

            ###3. FC layer & dropout
            with tf.name_scope("hidden_right"):
                xw_plus_b = tf.nn.xw_plus_b(self.h_pool_right, W, b, name="hidden_output_right")
                xw_plus_b_BN = self.batch_same(xw_plus_b, mean, var, beta, offset)
                self.hidden_output_right = tf.nn.tanh(xw_plus_b_BN)
                print(mean)
            with tf.name_scope("dropout_right"):
                self.h_drop_right = tf.nn.dropout(self.hidden_output_right, self.dropout_keep_prob,
                                                  name="hidden_output_drop_right")

        # Compute L1 distance
        with tf.name_scope("left_L1"):
            h = tf.reduce_sum(tf.abs(self.h_drop_left - self.h_drop_centre), 1)
            self.left_l1 = tf.negative(h)

        with tf.name_scope("right_L1"):
            h = tf.reduce_sum(tf.abs(self.h_drop_right - self.h_drop_centre), 1)
            self.right_l1 = tf.negative(h)
            print(self.right_l1)

        # Compute cosine
        print(self.h_drop_centre)
        print(self.h_drop_left)
        epsilon = 1e-5
        abs_centre = tf.sqrt(tf.reduce_sum(tf.square(self.h_drop_centre), 1))

        with tf.name_scope("cosine_left"):
            product = tf.reduce_sum(tf.multiply(self.h_drop_left, self.h_drop_centre), 1)
            abs_left = tf.sqrt(tf.reduce_sum(tf.square(self.h_drop_left), 1))
            self.cosine_left = product / tf.multiply(abs_left, abs_centre)
            # self.cosine_left = product / tf.maximum(tf.multiply(abs_left, abs_centre), epsilon)
            print(self.cosine_left)
        with tf.name_scope("cosine_right"):
            product = tf.reduce_sum(tf.multiply(self.h_drop_right, self.h_drop_centre), 1)
            abs_right = tf.sqrt(tf.reduce_sum(tf.square(self.h_drop_right), 1))
            self.cosine_right = product / tf.multiply(abs_right, abs_centre)
            print(self.cosine_right)

        # Softmax
        with tf.name_scope("softmax"):
            self.softmax_score = tf.nn.softmax(tf.stack([self.left_l1, self.right_l1], axis=1), name="score")
            print(self.softmax_score)
            self.predictions = tf.argmax(self.softmax_score, 1, name="predictions")

        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.stack, labels=self.input_y)
            losses = self.general_loss(logits=self.softmax_score, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            print("self.accuracy", self.accuracy.name)
