# coding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
from vocab_utils import Vocab
import numpy as np

max_len = 25
root_dir = "/home/haojianyong/file_1/CNN/training_models/nj_1/"


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


def build_input_data(data_left, data_centre, data_right, label, vocab):
    vocabset = set(vocab.keys())
    out_left = np.array(
        [[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence] for sentence in data_left])
    out_centre = np.array(
        [[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence] for sentence in data_centre])
    out_right = np.array(
        [[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence] for sentence in data_right])
    out_y = np.array(label)
    return [out_left, out_centre, out_right, out_y]


def batch_iter(all_data, batch_size, num_epochs, shuffle=False):
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


print("Loading data...")
wordVocab = Vocab()
wordVocab.fromText_format3("/home/haojianyong/file_1/CNN/", "data/wordvec.vec")
vocab_tuple = (wordVocab.word2id, wordVocab.id2word)

data_label = []
data_left = []
data_centre = []
data_right = []
for line in open("/home/haojianyong/file_1/CNN/data_dssm/tt.txt"):
    line = line.strip().strip("\n").split("\t")
    if len(line) != 4: continue
    data_label.append(map(int, line[0].split(" ")))
    data_left.append(line[1].strip().split(" "))
    data_centre.append(line[2].strip().split(" "))
    data_right.append(line[3].strip().split(" "))
x1 = data_left
x2 = data_centre
x3 = data_right
x_label = data_label
data_left = pad_sentences(data_left, max_len)
data_centre = pad_sentences(data_centre, max_len)
data_right = pad_sentences(data_right, max_len)
vocab, vocab_inv = vocab_tuple

x_left_dev, x_centre_dev, x_right_dev, y_dev = build_input_data(data_left, data_centre, data_right, data_label, vocab)

g_graph = tf.Graph()
with g_graph.as_default():
    with tf.gfile.GFile(root_dir + 'model_cnn_dssm.pb', "rb") as f:
        graph_def = tf.GraphDef()  # 先创建一个空的图
        graph_def.ParseFromString(f.read())  # 加载proto-buf中的模型
        tf.import_graph_def(graph_def, name='')  # 最后复制pre-def图的到默认图中

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        input_left = sess.graph.get_tensor_by_name("input_left:0")
        input_centre = sess.graph.get_tensor_by_name("input_centre:0")
        # input_right = sess.graph.get_tensor_by_name("input_right:0")
        # input_y = sess.graph.get_tensor_by_name("input_y:0")
        dropout_keep_prob = sess.graph.get_tensor_by_name("dropout_keep_prob:0")
        cosine_left = sess.graph.get_tensor_by_name("cosine_left/div:0")
        # cosine_right = sess.graph.get_tensor_by_name("cosine_right/div:0")
        # output_accuracy = sess.graph.get_tensor_by_name("accuracy/accuracy:0")
        graphMean = sess.graph.get_tensor_by_name("conv-maxpool-left-5/cond/Merge:0")

        softmax_score = sess.graph.get_tensor_by_name("softmax/score:0")

        pooled_outputs_left = sess.graph.get_tensor_by_name("conv-maxpool-left-2/pool:0")
        pooled_outputs_centre = sess.graph.get_tensor_by_name("conv-maxpool-centre-2/pool:0")
        # h_pool_left = sess.graph.get_tensor_by_name("h_pool_left:0")
        # h_pool_centre = sess.graph.get_tensor_by_name("h_pool_centre:0")
        # h_drop_centre = sess.graph.get_tensor_by_name("hidden_dropout/dropout_centre/hidden_output_drop_centre/mul:0")
        # h_drop_left = sess.graph.get_tensor_by_name("hidden_dropout/dropout_left/hidden_output_drop_left/mul:0")

        with open("/home/haojianyong/file_1/CNN/" + "data_dssm/result_nonmatch.txt", "w") as out_op:
            batches_dev, _ = batch_iter(list(zip(x_left_dev, x_centre_dev, x_right_dev, y_dev)), 64, num_epochs=1,
                                        shuffle=False)

            accuracies = []

            for idx, batch_dev in enumerate(batches_dev):
                x_left_batch_dev, x_centre_batch_dev, x_right_batch_dev, y_batch_dev = zip(*batch_dev)

                cosine1, Mean, pooled1, pooled2 = sess.run(
                    [cosine_left, graphMean,
                     pooled_outputs_left, pooled_outputs_centre],
                    feed_dict={input_left:x_left_batch_dev,
                               input_centre: x_centre_batch_dev,
                               # input_left: x_right_batch_dev,
                               # input_y: y_batch_dev,
                               dropout_keep_prob: 1.0
                               })

                # for i in range(len(y_batch_dev)):
                #     out_op.write(str(cosine1[0][i]) + "\n")

                # print("left: ")
                print(cosine1)
                # print("right: ")
                # print(cosine2)
                # print("curracy")
                # print(curracy)
                # print(Mean)
                # print(pooled1)
                # print(pooled2)
                print("---------------------")
                break
