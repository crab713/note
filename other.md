### BN

syncBN:在多张卡上同步BN，避免在单卡上bz=1，BN无法发挥作用的问题

[pytorch的坑](https://zhuanlan.zhihu.com/p/59271905)



### warmup

使用较低的learning rate预热，慢慢恢复到正常

[mmlab宝贝网站](https://openmmlab.com/)



### 模型训练次数指标

iter--一次迭代，指一个min_batch的一次forward+backward

epoch--指迭代完一次所有数据



### 基本概念

1. **降采样&&下采样**：指缩小图像，有最大值采样等等多种方法

2. **上采样&&图像插值**：指放大图像，在像素点之间插入新的元素

   最近邻插值法：将一格元素放大到周围几格，在上采样过程中最大程度保留特征图语义信息

   

3. 



#### MD的几个标识

```markdown
**这是加粗的文字**
*这是斜体文字* 

```

