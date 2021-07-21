### models.builder.py->build()

构建模型的根本方法

主Registry注册->models.builder.py

**流程**：从build_segmentor进入，传入cfg，使用encoderDecoder，再在该模型内逐个build各个组件（backbone等）。



### mmseg.apis.train.py->train_segmentor()

对模型进行训练的根本方法

74行：构建runner

91行：rigister hook



