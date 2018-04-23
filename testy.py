# coding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
from vocab_utils import Vocab
import numpy as np
import time

max_len_left = max_len_right = 25
root_dir="/home/haojianyong/file_1/pairCNN-Ranking-master"

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
            total.append(shuffled_data[start_index:end_index])
    return np.array(total), num_batches_per_epoch




print("Loading data...")
wordVocab = Vocab()
wordVocab.fromText_format3(root_dir, "data/wordvec.vec")
vocab_tuple = (wordVocab.word2id, wordVocab.id2word)

data_label = []
data_left = []
data_right = []

testPath = "/home/haojianyong/file_1/context_similarity/guangFaFAQ/guangFaFAQ_nonmatch_cut.txt"
for line in open(testPath):
    line = line.strip().strip("\n").split("\t")
    if len(line) < 3: continue
    data_label.append(int(line[0]))
    data_left.append(line[1].split(" "))
    data_right.append(line[2].split(" "))
x1 = data_left
x2 = data_right
x_label = data_label
data_left = pad_sentences(data_left, max_len_left)
data_right = pad_sentences(data_right, max_len_right)
vocab, vocab_inv = vocab_tuple

x_left_dev, x_right_dev, y_dev = build_input_data(data_left, data_right, data_label, vocab)

g_graph = tf.Graph()
with g_graph.as_default():
    with tf.gfile.GFile(root_dir+'/runs/model_cnn.pb', "rb") as f:
        graph_def = tf.GraphDef()  # 先创建一个空的图
        graph_def.ParseFromString(f.read())  # 加载proto-buf中的模型
        tf.import_graph_def(graph_def, name='')  # 最后复制pre-def图的到默认图中

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        input_left = sess.graph.get_tensor_by_name("input_left:0")
        input_right = sess.graph.get_tensor_by_name("input_right:0")
        input_y = sess.graph.get_tensor_by_name("input_y:0")
        dropout_keep_prob = sess.graph.get_tensor_by_name("dropout_keep_prob:0")
        output_accuracy = sess.graph.get_tensor_by_name("accuracy/accuracy:0")

        output_prob = sess.graph.get_tensor_by_name("output/prob:0")

        with open(root_dir+"/data/result_nonmatch.txt", "w") as out_op:
            t1 = time.time()
            batches_dev, _ = batch_iter(list(zip(x_left_dev, x_right_dev, y_dev)), 64, num_epochs=1, shuffle=False)

            accuracies = []

            for idx, batch_dev in enumerate(batches_dev):
                x_left_batch_dev, x_right_batch_dev, y_batch_dev = zip(*batch_dev)

                accuracy, prob = sess.run([output_accuracy, output_prob], feed_dict={input_left: x_left_batch_dev,
                                                                                     input_right: x_right_batch_dev,
                                                                                     input_y: y_batch_dev,
                                                                                     dropout_keep_prob: 1.0
                                                                                     })
                # accuracies.append(accuracy)
                # for i in range(len(y_batch_dev)):
                #     p = prob[i].tolist()
                #     y_label = y_batch_dev[i].tolist()
                #     if y_label[1] != [1 if p[1] >= 0.5 else 0][0]:
                #         result = str(y_label[1]) + "\t" + str(p[1]) + "\t" + " ".join(x1[i]) + "\t" + " ".join(
                #             x2[i]) + "\n"
                #         out_op.write(result)
            t2=time.time()
            print((t2-t1)/len(x_label)*1000,"ms")
        # print(np.mean(np.array(accuracies)))
