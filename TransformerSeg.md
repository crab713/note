### models.builder.py->build()

构建模型的根本方法

主Registry注册->models.builder.py

**流程**：从build_segmentor进入，传入cfg，使用encoderDecoder，再在该模型内逐个build各个组件（backbone等）。



### mmseg.apis.train.py->train_segmentor()

对模型进行训练的根本方法

74行：构建runner

91行：rigister hook



### train部分的output

在uper_head的forward中，模型最后经过了cls_seg方法，该方法为一个kernel_size=1的2D卷积，最后输出一个[batch, h, w, num_class]维度的Tensor



## SwinTransformer seg

#### 1. patch_embed

使用Conv2d对每个patch卷积，通道数即为embed_dim，stride=patch_size

在输入stage前embed，即输入第一个stage的embed为96，第二个为2*96

#### 2. PatchMerging

将相邻的四个patch合并为一个新的patch，通过Linear将四个dim合并映射为2*dim长度



## Inference部分

#### 1. 解析

在inference阶段，需要输入至少三个参数，分别为img，config，checkpoint，均为文件路径str，其中config为该checkpoint的配置文件，demo在demo文件夹中

基本用法为init_segmentor后，在调用inference_segmentor进行推断得到result，最后可使用show_result_pyplot函数对分割后图片进行展示

**inference_segmentor**:调用segmentors.base的forward方法，将return_loss参数设置为False，调用forward_test方法进行inference（实际调用实体类中的simple_test方法）

在inference_segmentor方法中，模型已经将img处理好，变换成[b,c=3,h,w]维度的Tensor，再传入model内inference

**EncoderDecoder.simple_test:**通过inference方法，获得模型softmax后的output存放于seg_logit，即每个像素的150个类的概率值，再通过argmax得到每个像素最可能的那个类存放于seg_pred，经过处理后，将seg_pred转换为list格式输出，维度为[h,w,1:which class]

#### 2. 输入准备

将文件名传入inference_segmentor即可，具体load_Image参数在模型配置文件中data.test部分定义

#### 3. 输出格式

根据simple_test中return的数据而定，不做变动则返回维度为[h,w,1]的list，最后一维代表该像素点的类别，可返回[h,w,channel]的原始output



