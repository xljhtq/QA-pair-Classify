# 此处用于cnn做语义相似度 and QA match
For pairCNN mdoel:
1.在similarity layer层中，假定了一个相似矩阵，让模型自学习 
2.在join layer层中，不仅把similarity因素进行concat,还把additional features也一起concat进来。
其中，additional features include: sentence lengths, another sample similarities (namely tf-idf) and so on , which depends on your project.
3.在最後的loss function中，通過cross_entropy_function和改造的hinge_loss_function比較，使用改造的loss函數進行訓練，可以得到相似的語句有更大的得分，更不相似的
語句有更小的得分。

Note:
1.在模型中，tf.nn.softmax_cross_entropy_with_logits包含兩個部分(softmax和交叉熵的計算)，所以當改造模型時候，必須先單獨使用softmax函數變成概率的取值，
否則在此之前是有負數的出現，干擾loss函數的計算

For DssmCnn model:
0.输入数据前, 先进行转成tfRecords文件格式方便内存的存储和迁移.
1.input是Query A. B. C, 输出是cosine(A,B) 和cosine(B,C), 最后再通过softmax进行判定属于哪一个query概率大
2.加入了BN层,同时当然是需要Moving Average值的辅助进行后续的预测的
3.加入了CHUNK Pooling层替代原有的Max Pooling层,可以部分解决因前后位置的不同导致的语义差异。
4.预测之时,可以只保留右边部分的模型pb文件,在保证性能的情况下进行预测