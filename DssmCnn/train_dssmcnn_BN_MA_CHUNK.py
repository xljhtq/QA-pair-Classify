#!/usr/bin/env python
# encoding=utf-8

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from rank_dssmcnn_BN_MA_CHUNK import Ranking_DSSMCNN
from vocab_utils import Vocab


def pad_sentence(sentence, sequence_length, padding_word="<PAD/>"):
    if len(sentence) < sequence_length:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
    else:
        new_sentence = sentence[:sequence_length]
    return new_sentence


def build_input_data(data, vocab, vocabset):
    out_data = [vocab[word] if word in vocabset else vocab['<UNK/>'] for word in data]
    return out_data


def load_data(filepath, vocab_tuple=None):
    vocab, vocab_inv = vocab_tuple
    vocabset = set(vocab.keys())
    data_label = []
    data_left = []
    data_centre = []
    data_right = []
    dic = {}
    i=0
    for line in open(filepath):
        line = line.strip().strip("\n").split("\t")
        i+=1
        if len(line) != 4:
            print(i)
            continue
        data_label.append([int(x) for x in line[0].split(" ")])

        leftList = line[1].strip().split(" ")
        leftSentence = pad_sentence(leftList, FLAGS.max_len)
        out_left = build_input_data(leftSentence, vocab, vocabset)
        data_left.append(out_left)

        centreList = line[2].strip().split(" ")
        centreSentence = pad_sentence(centreList, FLAGS.max_len)
        out_centre = build_input_data(centreSentence, vocab, vocabset)
        data_centre.append(out_centre)

        rightLsit = line[3].strip().split(" ")
        rightSentence = pad_sentence(rightLsit, FLAGS.max_len)
        out_right = build_input_data(rightSentence, vocab, vocabset)
        data_right.append(out_right)

        for word in leftList:
            if word not in dic:
                dic[word] = 1
        for word in centreList:
            if word not in dic:
                dic[word] = 1
        for word in rightLsit:
            if word not in dic:
                dic[word] = 1
    print("word dic length: ", len(dic))
    data_left = np.array(data_left)
    data_centre = np.array(data_centre)
    data_right = np.array(data_right)
    out_y = np.array(data_label)
    return data_left, data_centre, data_right, out_y, vocab, vocab_inv


def main(_):
    print(FLAGS)

    print("Loading data...")
    wordVocab = Vocab()
    wordVocab.fromText_format3(FLAGS.train_dir, FLAGS.wordvec_path)
    vocab_tuple = (wordVocab.word2id, wordVocab.id2word)
    x_left_train, x_centre_train, x_right_train, y_train_label, vocab, vocab_inv = load_data(
        os.path.join(FLAGS.train_dir, FLAGS.train_path), vocab_tuple=vocab_tuple)

    x_left_dev, x_centre_dev, x_right_dev, y_dev, vocab, vocab_inv = load_data(
        os.path.join(FLAGS.train_dir, FLAGS.test_path), vocab_tuple=vocab_tuple)

    print("Loading Model...")
    sys.stdout.flush()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = Ranking_DSSMCNN(
                max_len=FLAGS.max_len,
                vocab_size=len(vocab),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                num_hidden=FLAGS.num_hidden,
                fix_word_vec=FLAGS.fix_word_vec,
                word_vocab=wordVocab,
                C=4,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            sess.run(tf.global_variables_initializer())

            has_pre_trained_model = False
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs"))

            print(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            else:
                print("continue training models")
                ckpt = tf.train.get_checkpoint_state(out_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("-------has_pre_trained_model--------")
                    print(ckpt.model_checkpoint_path)
                    has_pre_trained_model = True
                    sys.stdout.flush()

            checkpoint_prefix = os.path.join(out_dir, "model")
            if has_pre_trained_model:
                print("Restoring model from " + ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("DONE!")
                sys.stdout.flush()

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

            def dev_whole(x_left_dev, x_centre_dev, x_right_dev, y_dev):
                batches_dev, _ = batch_iter(
                    list(zip(x_left_dev, x_centre_dev, x_right_dev, y_dev)), FLAGS.batch_size * 2, num_epochs=1,
                    shuffle=False)
                losses = []
                accuracies = []
                for idx, batch_dev in enumerate(batches_dev):
                    x_left_batch_dev, x_centre_batch_dev, x_right_batch_dev, y_batch_dev = zip(*batch_dev)
                    feed_dict = {
                        cnn.input_left: x_left_batch_dev,
                        cnn.input_centre: x_centre_batch_dev,
                        cnn.input_right: x_right_batch_dev,
                        cnn.input_y: y_batch_dev,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, loss, accuracy, softmax_score, left, right = sess.run(
                        [global_step, cnn.loss, cnn.accuracy, cnn.softmax_score, cnn.cosine_left, cnn.cosine_right],
                        feed_dict)
                    losses.append(loss)
                    accuracies.append(accuracy)
                print("left_cosine: ", left)
                print("right_cosine: ", right)
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
            batches_epochs, num_batches_per_epoch = batch_iter(
                list(zip(x_left_train, x_centre_train, x_right_train, y_train_label)),
                FLAGS.batch_size,
                FLAGS.num_epochs, shuffle=False)

            print(num_batches_per_epoch)
            dev_accuracy = []
            train_loss = 0
            total_loss = []
            for batch in batches_epochs:
                x1_batch, x_centre_batch, x2_batch, y_batch = zip(*batch)
                feed_dict = {
                    cnn.input_left: x1_batch,
                    cnn.input_centre: x_centre_batch,
                    cnn.input_right: x2_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, current_step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                train_loss += loss

                if current_step % 10000 == 0:
                    print("step {}, loss {}, acc {}".format(current_step, loss, accuracy))
                    sys.stdout.flush()

                if (current_step + 1) % num_batches_per_epoch == 0 or (
                        current_step + 1) == num_batches_per_epoch * FLAGS.num_epochs:
                    print((current_step + 1) / num_batches_per_epoch, " epoch, train_loss:", train_loss)
                    total_loss.append(train_loss)
                    train_loss = 0
                    sys.stdout.flush()

                if (current_step + 1) % num_batches_per_epoch == 0 or (
                        current_step + 1) == num_batches_per_epoch * FLAGS.num_epochs:
                    print("\nEvaluation:")
                    loss, accuracy = dev_whole(x_left_dev, x_centre_dev, x_right_dev, y_dev)
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

                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                                    output_node_names=[
                                                                                        'accuracy/accuracy',
                                                                                        'softmax/score',
                                                                                        'cosine_left/div',
                                                                                        'cosine_right/div'])
                    for node in output_graph_def.node:
                        if node.op == 'RefSwitch':
                            node.op = 'Switch'
                            for index in xrange(len(node.input)):
                                if 'moving_' in node.input[index]:
                                    node.input[index] = node.input[index] + '/read'
                        elif node.op == 'AssignSub':
                            node.op = 'Sub'
                            if 'use_locking' in node.attr: del node.attr['use_locking']
                    with tf.gfile.GFile(FLAGS.train_dir + "runs/model_cnn_dssm.pb", "wb") as f:
                        f.write(output_graph_def.SerializeToString())
                    print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordvec_path", default="data/wordvec.vec", help="wordvec_path")
    parser.add_argument("--train_dir", default="/home/haojianyong/file_1/CNN/", help="Training dir root")
    parser.add_argument("--train_path", default="data_dssm/train.txt", help="train path")
    parser.add_argument("--test_path", default="data_dssm/test.txt", help="test path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs (default: 200)")
    parser.add_argument("--max_len", type=int, default=25, help="max document length of input")
    parser.add_argument("--fix_word_vec", default=True, help="fix_word_vec")

    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Dimensionality of character embedding (default: 64)")
    parser.add_argument("--filter_sizes", default="2,3,4,5", help="Comma-separated filter sizes (default: '2,3')")
    parser.add_argument("--num_filters", type=int, default=100, help="Number of filters per filter size (default: 64)")
    parser.add_argument("--num_hidden", type=int, default=100, help="Number of hidden layer units (default: 100)")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", type=float, default=1e-4, help="L2 regularizaion lambda")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="L2 regularizaion lambda")
    parser.add_argument("--allow_soft_placement", default=True, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False, help="Log placement of ops on devices")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
