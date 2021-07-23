### BN

syncBN:在多张卡上同步BN，避免在单卡上bz=1，BN无法发挥作用的问题

[pytorch的坑](https://zhuanlan.zhihu.com/p/59271905)



### warmup

使用较低的learning rate预热，慢慢恢复到正常

[mmlab宝贝网站](https://openmmlab.com/)



### 模型训练次数指标

iter--一次迭代，指一个min_batch的一次forward+backward

epoch--指迭代完一次所有数据



### Conv layer扩边处理

1. 所有的conv层都是：kernel_size=3，pad=1，stride=1
2. 所有的pooling层都是：kernel_size=2，pad=0，stride=2

对卷积做扩边处理，原图扩大使得卷积过后不改变输入和输出矩阵大小，只有pooling层才使矩阵长宽变为原来的1/2



### PPM金字塔池化模型

可以在不同的尺度下来保存全局信息，比简单的单一pooling更能保留全局上下文信息。

标准过程：对特征图分别池化到目标size（不同卷积核），进行卷积将channel减少到原来的1/N，再通过上采样得到相同size，concat拼接，使特征图和channel不变。



### ReLU系列激活函数

leakyReLU：稍微解决神经元“死亡”的问题，近似于ReLU，但在输入小于0的部分，值为负，并且有微小的梯度，α取值一般为0.01

PReLU激活函数：在LeakyReLU的基础上，将α作为需要学习的参数

ELU：输入大于0的部分梯度为1，输入小于0的部分无限趋近于-α，兼顾两者优点



### 基本概念

1. **降采样&&下采样**：指缩小图像，有最大值采样等等多种方法

2. **上采样&&图像插值**：指放大图像，在像素点之间插入新的元素

   最近邻插值法：将一格元素放大到周围几格，在上采样过程中最大程度保留特征图语义信息

3. **mIOU**：mIOU的定义：计算真实值和预测值两个集合的交集和并集之比。这个比例可以变形为TP（交集）比上TP、FP、FN之和（并集）。即：mIOU=TP/(FP+FN+TP)。

4. **Pixel Accuracy**（PA，像素精度）：即标记正确的像素占总像素的比例。

5. **Mean Pixel Accuracy**（MPA，平均像素精度）：每个类内被正确分类像素数的比例，之后求所有类的平均。

6. **感受野**：卷积神经网络每一层输出的特征图上的像素点在输入图片上映射的区域大小。

7. **梯度消失**：极小的梯度无法让参数得到有效更新，即使有微小的更新，浅层和深层网络参数的更新速率也相差巨大。

8. **Hook**：处理被拦截的函数调用，消息等的代码，实现可通过传入一个函数的方式，函数即为一个Hook。pytorch中每一层也可通过register_forward_hook来使用。

9. **resume**：用于训练过程中停止然后恢复训练时加载的权重，与checkpoint类似但不相同

10. **Image Matting**：在segmention的基础上，对图像抠图，形成三元组trimap，处理获得更细致的前景图像



#### MD的几个标识

```markdown
**这是加粗的文字**
*这是斜体文字* 

```

### inspect

提供了一些有用的函数帮助获取对象的信息[inspect使用文档](https://docs.python.org/zh-cn/3.7/library/inspect.html)

例：inspect.isclass(obj) 如果obj是类返回True



## SwinTransformer seg

#### 1. patch_embed

使用Conv2d对每个patch卷积，通道数即为embed_dim，stride=patch_size

在输入stage前embed，即输入第一个stage的embed为96，第二个为2*96

#### 2. PatchMerging

将相邻的四个patch合并为一个新的patch，通过Linear将四个dim合并映射为2*dim长度



### paper解析

1. [Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)
2. [FPN](https://zhuanlan.zhihu.com/p/148738276)
3. [FCN](https://blog.csdn.net/qq_36269513/article/details/80420363)
4. [UPerNet](https://www.cnblogs.com/alan-blog-TsingHua/p/9736167.html)

### 解析文章

1. [mmseg解析](https://blog.csdn.net/qq_32425195/article/details/110392397)

2. [mm系模型使用](https://zhuanlan.zhihu.com/p/163747610)

3. [mmlab官方解析](https://www.zhihu.com/people/openmmlab/)

4. [pytorch源码解读索引贴](https://zhuanlan.zhihu.com/p/328674159)

