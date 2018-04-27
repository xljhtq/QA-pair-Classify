# 此处用于cnn做语义相似度 or QA match
关键点： 
1.在similarity layer层中，假定了一个相似矩阵，让模型自学习 
2.在join layer层中，不仅把similarity因素进行concat,还把additional features也一起concat进来。
其中，additional features include: sentence lengths, another sample similarities (namely tfidf) and so on , which depends on your project.
3.在最後的loss function中，通過cross_entropy_function和改造的hinge_loss_function比較，使用改造的loss函數進行訓練，可以得到相似的語句有更大的得分，更不相似的
語句有更小的得分。

Note:
1.在模型中，tf.nn.softmax_cross_entropy_with_logits包含兩個部分(softmax和交叉熵的計算)，所以當改造模型時候，必須先單獨使用softmax函數變成概率的取值，
否則在此之前是有負數的出現，干擾loss函數的計算