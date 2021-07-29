### 模型val要点

**mIoU**位置：mmseg.core.evaluation.metrics.py



# 7.23任务

### 1.编写验证部分代码

使用Inference_demo慢慢跑吧。。。



### 2.数据集

### compositon-1k

该数据集基于COCO和VOC数据集，通过网络生成得到，该数据集包含49300个训练图片和1000个测试图片

该数据集的生成方法大致为通过分离前景再融合到其它背景上得到

数据集介绍：
https://paperswithcode.com/dataset/composition-1k
https://paperswithcode.com/dataset/bg-20k
数据集获取：
https://mmediting.readthedocs.io/en/latest/matting_datasets.html
https://github.com/open-mmlab/mmediting/blob/master/docs/getting_started.md

```ruby

```

### COCO&VOC

数据集下载链接：https://zhuanlan.zhihu.com/p/340900096

VOC镜像：https://pjreddie.com/projects/pascal-voc-dataset-mirror/

各种数据集介绍：https://blog.csdn.net/u010189457/article/details/78472550

VOC数据集说明：https://blog.csdn.net/haoji007/article/details/80361587

### 3. 反卷积方法


