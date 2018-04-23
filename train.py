#!/usr/bin/env python
# encoding=utf-8

import os
import sys
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter
from rank import Ranking
from vocab_utils import Vocab


def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_input_data(data_left, data_right, label, vocab):
    vocabset = set(vocab.keys())
    out_left = np.array(
        [[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence] for sentence in data_left])
    out_right = np.array(
        [[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence] for sentence in data_right])
    out_y = np.array([[0, 1] if x == 1 else [1, 0] for x in label])
    return [out_left, out_right, out_y]


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    # id2word
    t = word_counts.most_common(FLAGS.most_words - 1)  # 根据出现次数倒序排序
    vocabulary_inv = [x[0] for x in t]  # 目的只是建立词表
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv.append('<UNK/>')
    # word2id
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def load_data(filepath, vocab_tuple=None):
    data_label = []
    data_left = []
    data_right = []
    for line in open(filepath):
        line = line.strip().strip("\n").split("\t")
        if len(line) < 3: continue
        data_label.append(int(line[0]))
        data_left.append(line[1].split(" "))
        data_right.append(line[2].split(" "))

    num_pos = sum(data_label)
    # 需要优化
    data_left = pad_sentences(data_left, FLAGS.max_len_left)
    data_right = pad_sentences(data_right, FLAGS.max_len_right)

    if vocab_tuple is None:
        vocab, vocab_inv = build_vocab(data_left + data_right)
    else:
        vocab, vocab_inv = vocab_tuple

    data_left, data_right, data_label = build_input_data(data_left, data_right, data_label, vocab)
    return data_left, data_right, data_label, vocab, vocab_inv, num_pos


def main(_):
    # Load data
    print("Loading data...")
    wordVocab = Vocab()
    wordVocab.fromText_format3(FLAGS.train_dir, FLAGS.wordvec_path)
    vocab_tuple = (wordVocab.word2id, wordVocab.id2word)
    x_left_train, x_right_train, y_train_label, vocab, vocab_inv, num_pos = load_data(
        os.path.join(FLAGS.train_dir, 'data/train.txt'), vocab_tuple=vocab_tuple)

    x_left_dev, x_right_dev, y_dev, vocab, vocab_inv, num_pos = load_data(
        os.path.join(FLAGS.train_dir, 'data/test.txt'), vocab_tuple=vocab_tuple)

    print("Loading Model...")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = Ranking(
                max_len_left=FLAGS.max_len_left,
                max_len_right=FLAGS.max_len_right,
                vocab_size=len(vocab),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                num_hidden=FLAGS.num_hidden,
                fix_word_vec=FLAGS.fix_word_vec,
                word_vocab=wordVocab,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            sess.run(tf.global_variables_initializer())

            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs"))
            print(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            else:
                print("delete runs/")
                tf.gfile.DeleteRecursively(out_dir)
                os.makedirs(out_dir)

            print("Writing to {}\n".format(out_dir))
            checkpoint_prefix = os.path.join(out_dir, "model")

            def batch_iter(all_data, batch_size, num_epochs, shuffle=True):
                total = []
                data = np.array(all_data)
                data_size = len(data)
                num_batches_per_epoch = int(data_size / batch_size) + 1
                for epoch in range(num_epochs):
                    if shuffle:
                        shuffle_indices = np.random.permutation(np.arange(data_size))
                        shuffled_data = data[shuffle_indices]
                    else:
                        shuffled_data = data

                    for batch_num in range(num_batches_per_epoch):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size)
                        # yield shuffled_data[start_index:end_index]
                        total.append(shuffled_data[start_index:end_index])
                return np.array(total), num_batches_per_epoch

            def dev_whole(x_left_dev, x_right_dev, y_dev):
                batches_dev, _ = batch_iter(
                    list(zip(x_left_dev, x_right_dev, y_dev)), FLAGS.batch_size * 2, num_epochs=1, shuffle=False)
                losses = []
                accuracies = []
                for idx, batch_dev in enumerate(batches_dev):
                    x_left_batch_dev, x_right_batch_dev, y_batch_dev = zip(*batch_dev)
                    feed_dict = {
                        cnn.input_left: x_left_batch_dev,
                        cnn.input_right: x_right_batch_dev,
                        cnn.input_y: y_batch_dev,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy, sims, prob = sess.run(
                        [global_step, cnn.loss, cnn.accuracy, cnn.sims, cnn.prob],
                        feed_dict)
                    losses.append(loss)
                    accuracies.append(accuracy)
                return np.mean(np.array(losses)), np.mean(np.array(accuracies))

            def overfit(dev_loss):
                n = len(dev_loss)
                if n < 4:
                    return False
                for i in range(n - 4, n):
                    if dev_loss[i] > dev_loss[i - 1]:
                        return False
                return True

            # Generate batches
            batches_epochs, num_batches_per_epoch = batch_iter(list(zip(x_left_train, x_right_train, y_train_label)),
                                                               FLAGS.batch_size,
                                                               FLAGS.num_epochs)

            # Training loop. For each batch...
            dev_accuracy = []
            train_loss = 0
            total_loss = []
            for batch in batches_epochs:
                x1_batch, x2_batch, y_batch = zip(*batch)
                feed_dict = {
                    cnn.input_left: x1_batch,
                    cnn.input_right: x2_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, current_step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                train_loss += loss

                if current_step % 10000 == 0:
                    print("step {}, loss {:g}, acc {:g}".format(current_step, loss, accuracy))
                    sys.stdout.flush()

                if (current_step + 1) % num_batches_per_epoch == 0 or (
                        current_step + 1) == num_batches_per_epoch * FLAGS.num_epochs:
                    print("One epoch, train_loss:", train_loss)
                    total_loss.append(train_loss)
                    train_loss = 0
                    sys.stdout.flush()

                if (current_step + 1) % num_batches_per_epoch == 0 or (
                        current_step + 1) == num_batches_per_epoch * FLAGS.num_epochs:
                    print("\nEvaluation:")
                    loss, accuracy = dev_whole(x_left_dev, x_right_dev, y_dev)
                    dev_accuracy.append(accuracy)
                    print("Recently accuracy:")
                    print (dev_accuracy[-10:])
                    print("Recently train_loss:")
                    print(total_loss[-10:])
                    if overfit(dev_accuracy):
                        print ('Overfit!!')
                        break
                    print("")
                    sys.stdout.flush()

                if (current_step + 1) % num_batches_per_epoch == 0 or (
                        current_step + 1) == num_batches_per_epoch * FLAGS.num_epochs:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                    sys.stdout.flush()


if __name__ == '__main__':
    # Parameters
    # ==================================================

    # Model Hyperparameters
    # To modify
    tf.flags.DEFINE_string("wordvec_path", "data/wordvec.vec", "wordvec_path")
    tf.flags.DEFINE_string("train_dir", "/home/haojianyong/file_1/pairCNN-Ranking-master/", "Training dir root")
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("max_len_left", 25, "max document length of left input")
    tf.flags.DEFINE_integer("max_len_right", 25, "max document length of right input")
    tf.flags.DEFINE_boolean("fix_word_vec", True, "fix_word_vec")

    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 64)")
    tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '2,3')")
    tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
    tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden layer units (default: 100)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 1e-4, "L2 regularizaion lambda (default: 0.0)")
    tf.flags.DEFINE_integer("most_words", 300000, "Most number of words in vocab (default: 300000)")
    # Training parameters
    tf.flags.DEFINE_integer("seed", 123, "Random seed (default: 123)")
    tf.flags.DEFINE_float("eval_split", 0.1, "Use how much data for evaluating (default: 0.1)")
    tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    tf.app.run()
