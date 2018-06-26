#!/usr/bin/env python
# encoding=utf-8

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from rank_dssmcnn_BN_MA_CHUNK import Ranking_DSSMCNN
from vocab_utils_tfrecords import Vocab


def countLines(train_path):
    totalLines = 0
    for line in open(train_path):
        totalLines += 1
    return totalLines


def processTFrecords(savePath, fileNumber=2):
    def pad_sentence(sentence, sequence_length=25, padding_word="<UNK/>"):
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        return new_sentence

    totalLines = 0
    for i in range(fileNumber):
        openFileName = savePath + "train." + str(i)
        for line in open(openFileName):
            line = line.strip().strip("\n").split("\t")
            if len(line) != 4: continue
            totalLines += 1

    return totalLines


def read_records(filenameList, max_len=25, epochs=30, batch_size=128):
    train_queue = tf.train.string_input_producer(filenameList, shuffle=True, num_epochs=epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(train_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.VarLenFeature(tf.int64),
            'query1': tf.VarLenFeature(tf.int64),
            'query2': tf.VarLenFeature(tf.int64),
            'query3': tf.VarLenFeature(tf.int64)

        })

    label = tf.sparse_tensor_to_dense(features['label'])
    query1 = tf.sparse_tensor_to_dense(features['query1'])
    query2 = tf.sparse_tensor_to_dense(features['query2'])
    query3 = tf.sparse_tensor_to_dense(features['query3'])

    label = tf.cast(label, tf.float32)
    query1 = tf.cast(query1, tf.int32)
    query2 = tf.cast(query2, tf.int32)
    query3 = tf.cast(query3, tf.int32)

    label = tf.reshape(label, [2])
    query1 = tf.reshape(query1, [max_len])
    query2 = tf.reshape(query2, [max_len])
    query3 = tf.reshape(query3, [max_len])

    # query1_batch_serialized, query2_batch_serialized, query3_batch_serialized, label_batch = tf.train.shuffle_batch(
    #     [query1, query2, query3, label], batch_size=batch_size,
    #     num_threads=4,
    #     capacity=10000 + 8 * batch_size,
    #     min_after_dequeue=10000)
    query1_batch_serialized, query2_batch_serialized, query3_batch_serialized, label_batch = tf.train.batch(
        [query1, query2, query3, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=10000 + 8 * batch_size)

    return query1_batch_serialized, query2_batch_serialized, query3_batch_serialized, label_batch


def processTFrecords_test(test_path, fileNumber=1):
    def pad_sentence(sentence, sequence_length=25, padding_word="<UNK/>"):
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        return new_sentence

    totalLines = 0
    for i in range(fileNumber):
        openFileName = test_path
        for line in open(openFileName):
            line = line.strip().strip("\n").split("\t")
            if len(line) != 4: continue
            totalLines += 1

    return totalLines


def main(_):
    print(FLAGS)
    save_path = FLAGS.train_dir + "tfFile/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("2.Loading WordVocab data...")
    wordVocab = Vocab()
    wordVocab.fromText_format3(FLAGS.train_dir, FLAGS.wordvec_path)
    sys.stdout.flush()

    # print("3.1 Start generating TFrecords File--train...")
    # # totalLines_train = processTFrecords(savePath=save_path, fileNumber=FLAGS.fileNumber)
    # print("totalLines_train:", totalLines_train)
    # sys.stdout.flush()
    # print("3.2 Start generating TFrecords File--test...")
    # test_path = FLAGS.train_dir + FLAGS.test_path
    # totalLines_test = processTFrecords_test(test_path=test_path, fileNumber=1)
    # print("totalLines_test:", totalLines_test)
    # sys.stdout.flush()

    print("4.Start loading TFrecords File...")
    fileNameList = []
    for i in range(FLAGS.fileNumber):
        string = FLAGS.train_dir + 'tfFile/train-' + str(i) + '.tfrecords'
        fileNameList.append(string)
    print("fileNameList: ", fileNameList)
    sys.stdout.flush()

    with tf.Graph().as_default():

        print("Loading Model...")
        sys.stdout.flush()

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("------------train model--------------")
            # with tf.variable_scope("Model", reuse=None):
            cnn = Ranking_DSSMCNN(
                max_len=FLAGS.max_len,
                vocab_size=len(wordVocab.word2id),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                num_hidden=FLAGS.num_hidden,
                fix_word_vec=FLAGS.fix_word_vec,
                word_vocab=wordVocab,
                C=4,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

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

            if has_pre_trained_model:
                print("Restoring model from " + ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("DONE!")
                sys.stdout.flush()

            print("-------------------Saved model checkpoint to--------------------")
            sys.stdout.flush()
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names=['cosine_right/div'])
            for node in output_graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in xrange(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
            with tf.gfile.GFile(FLAGS.train_dir + "runs/model_cnn_dssm_test2.pb", "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph.\n" % len(output_graph_def.node))

            sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordvec_path", default="data/wordvec.vec", help="wordvec_path")
    parser.add_argument("--train_dir", default="/home/haojianyong/file_1/CNN/", help="Training dir root")
    parser.add_argument("--train_path", default="data_dssm/train.txt", help="train path")
    parser.add_argument("--test_path", default="data_dssm/train2.txt", help="test path")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch Size (default: 64)")
    parser.add_argument("--fileNumber", type=int, default=2, help="Number of tfRecordsfile (default: 5)")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs (default: 200)")
    parser.add_argument("--max_len", type=int, default=25, help="max document length of input")
    parser.add_argument("--fix_word_vec", default=True, help="fix_word_vec")

    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Dimensionality of character embedding (default: 64)")
    parser.add_argument("--filter_sizes", default="2,3", help="Comma-separated filter sizes (default: '2,3')")
    parser.add_argument("--num_filters", type=int, default=100, help="Number of filters per filter size (default: 64)")
    parser.add_argument("--num_hidden", type=int, default=100, help="Number of hidden layer units (default: 100)")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", type=float, default=1e-4, help="L2 regularizaion lambda")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="L2 regularizaion lambda")
    parser.add_argument("--allow_soft_placement", default=True, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False, help="Log placement of ops on devices")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
