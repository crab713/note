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

11. **ground truth**：指人工标记的，真实的标记label



### ArgumentParser

``` python 
import argparse
# 创建一个解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--seed', type=int, default=72, help='Random seed.') # 名称前加--为可选参数
parser.add_argument('model_path', default='hello') # 名称前无--为必选参数

# 解析参数
args = parser.parse_args()
args.model_path # 参数使用
```



### 图像缩放

在模型训练中，通常都需要缩放图片到某一尺寸，这时使用pytorch或者numpy中的resize方法会使图像缩放失真，宜使用**PIL.Image**模块进行处理，或使用**cv2**也可

```python
# Image
im = Image.open(origin_path+image_name)
im = im.resize((512,512),Image.ANTIALIAS)

# CV2
img = cv2.resize(src=img, dsize=(1024, 768), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
```



### 图像导出

使用cv2.imwrite()

``` python
cv2.imwrite(filename, img)
```





### loss的设计

loss函数是用来对模型进行评估的函数，其值需要能反映出模型预测的表现程度，常见的有均方差MSE和交叉熵CrossEntropy，但MSE函数在分类问题中，使用softmax得到概率后再采用梯度下降法进行学习，会出现概率接近0或1时，梯度近似于0，学习速率非常慢的问题，因此再分类问题中优选loss为凸函数的形式。

CrossEntropy的优点在于它是一个凸函数（log），再模型训练时学习效率比较快，同时更容易找到全局最优值

同时如同交叉熵这种取log的损失函数好在能把一些函数之间的乘法变成函数之间的加法，对于凸函数而言，凸函数加凸函数依然是凸函数，更有利于多类别的训练，防止乘法溢出

**nn.L1Loss**:取预测值和真实值的绝对误差平均数

**nn.SmoothL1Loss**:误差在(-1, 1)取平方，其他时候是L1



### loss出现nan

在训练中途遇到一张噪点极多的图片，使得损失值在计算过程中数值溢出，该错误有可能是由于损失函数计算中的pow()操作所导致，这时可考虑换用abs()代替，或找其他方法



### BN

syncBN:在多张卡上同步BN，避免在单卡上bz=1，BN无法发挥作用的问题



### warmup

使用较低的learning rate预热，慢慢恢复到正常，该方法有可能能提高模型的准确率



### 模型训练次数指标

iter--一次迭代，指一个min_batch的一次forward+backward

epoch--指迭代完一次所有数据



### ReLU系列激活函数

leakyReLU：稍微解决神经元“死亡”的问题，近似于ReLU，但在输入小于0的部分，值为负，并且有微小的梯度，α取值一般为0.01

PReLU激活函数：在LeakyReLU的基础上，将α作为需要学习的参数

ELU：输入大于0的部分梯度为1，输入小于0的部分无限趋近于-α，兼顾两者优点



### Softmax

用来将向量中的值映射为一个概率值，并相加等于1

在映射过程中，较大的值所映射得到的概率值会相对偏大，较小的值会映射在一个较密集的区域，且概率值较小

**注意点**：在训练过程，如果模型中加入softmax方法，容易导致模型过于趋近于较大的那几个值，导致模型训练偏向极端方向，因此在训练过程中加入softmax不一定适用，故通常用于二分类中。

在该方法与交叉熵同时使用时，容易学习类间的信息，因为采用了类间竞争机制，但却忽略了其他非正确标签的差异，导致学习到的特征比较散。



### Padding

图像在送入模型训练前，通常都需要通过resize和padding操作来适应模型大小，但在这过程中padding的值选择不当会导致模型训练出现问题，如标签只有0，1，2时，选择padding值为255，故会导致训练出现严重偏移



### Conv layer扩边处理

1. 所有的conv层都是：kernel_size=3，pad=1，stride=1
2. 所有的pooling层都是：kernel_size=2，pad=0，stride=2

对卷积做扩边处理，原图扩大使得卷积过后不改变输入和输出矩阵大小，只有pooling层才使矩阵长宽变为原来的1/2



### 学习器

#### SGD

最经典的学习算法，没有动量的概念，是最简单的梯度下降算法。

$$n_t = a * g_t $$

最大缺点在于下降速度慢，可能会在沟壑两边持续震荡，停留在局部最优点

#### Adam

同时使用了一阶动量与二阶动量，二阶动量是固定时间窗口内的积累，随着时间窗口的变化，遇到的数据可能发生巨变，在训练后期引起学习率的震荡，导致模型无法收敛。





### PPM金字塔池化模型

可以在不同的尺度下来保存全局信息，比简单的单一pooling更能保留全局上下文信息。

标准过程：对特征图分别池化到目标size（不同卷积核），进行卷积将channel减少到原来的1/N，再通过上采样得到相同size，concat拼接，使特征图和channel不变。



### 深度估计

从单张2D图像中估计出物体所处的深度，知乎介绍链接：[大致介绍](https://zhuanlan.zhihu.com/p/29864012)

2014大致方法：卷积后上采样，分pretrain和fine两部分，最后得到每个pixel的深度值



### 巧妙将np.array转换为三值标签

``` python
def trivalurize(M):
	return (M>=50).astype(int)-(M<=150).astype(int)+1
```




#### MD的几个标识

```markdown
**这是加粗的文字**
*这是斜体文字* 

```

### inspect

提供了一些有用的函数帮助获取对象的信息[inspect使用文档](https://docs.python.org/zh-cn/3.7/library/inspect.html)

例：inspect.isclass(obj) 如果obj是类返回True



### paper解析

1. [Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)
2. [FPN](https://zhuanlan.zhihu.com/p/148738276)
3. [FCN](https://blog.csdn.net/qq_36269513/article/details/80420363)
4. [UPerNet](https://www.cnblogs.com/alan-blog-TsingHua/p/9736167.html)
5. [UNet](https://blog.csdn.net/weixin_45074568/article/details/114901600)

### 解析文章

1. [mmseg解析](https://blog.csdn.net/qq_32425195/article/details/110392397)
2. [mm系模型使用](https://zhuanlan.zhihu.com/p/163747610)
3. [mmlab官方解析](https://www.zhihu.com/people/openmmlab/)
4. [pytorch源码解读索引贴](https://zhuanlan.zhihu.com/p/328674159)
5. [交叉熵解析](https://zhuanlan.zhihu.com/p/35709485)
6. [transform应用](https://blog.csdn.net/qq_35027690/article/details/103742697)
7. [基础的loss函数](https://zhuanlan.zhihu.com/p/81956896)
8. [HDF5文件读写](https://blog.csdn.net/tracelessle/article/details/108304011)


### 网站杂

1. [pytorch的坑](https://zhuanlan.zhihu.com/p/59271905)
2. [mmlab坑人网站](https://openmmlab.com/)
3. [paperwithcode](https://paperswithcode.com/)
4. [DL基础知识](http://zh.gluon.ai/chapter_computer-vision/ssd.html)

