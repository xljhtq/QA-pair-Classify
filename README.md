# 此处用于cnn做语义相似度 or QA match
关键点： 
1.在similarity layer层中，假定了一个相似矩阵，让模型自学习 
2.在join layer层中，不仅把similarity因素进行concat,还把additional features也一起concat进来。
其中，additional features include: sentence lengths, another sample similarities (namely tfidf) and so on , which depends on your project.
