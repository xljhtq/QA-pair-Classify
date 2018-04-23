#coding=utf8
import argparse
import tensorflow as tf

def freeze_graph(ckpt_dictionary):
    checkpoint = tf.train.get_checkpoint_state(ckpt_dictionary)  # 检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path          # 得ckpt文件路径
    print input_checkpoint
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)  # 得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['accuracy/accuracy','output/prob'])
        with tf.gfile.GFile("/home/haojianyong/file_1/pairCNN-Ranking-master/runs/model_cnn.pb", "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dictionary", type=str, default="/home/haojianyong/file_1/pairCNN-Ranking-master/runs",help="input ckpt model dir")
    args = parser.parse_args()
    print(args.ckpt_dictionary)
    freeze_graph(args.ckpt_dictionary)