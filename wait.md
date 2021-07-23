collate_fn(nn.utils.DataLoader中、诡异bug中)







**诡异bug**:

mmseg.datasets.builder.py 145行 -> mmcv->parallel->collate.py batch=1



# 7.23任务

### 1.编写验证部分代码

可能是工作流未设置正确：在config中，未设置no-validate，在train_segmentor中依旧注册了eval hook，但workflow只有train一个工作流，也没载入val_dataset，出现bug

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

文件构成

数据集下载链接：https://zhuanlan.zhihu.com/p/340900096



### 3. 反卷积方法



### 4. 额外lgy ocr

### The Chars74K dataset

**数据说明**：https://www.cnblogs.com/qdsclove/p/5865463.html

**下载链接**：http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/#download

### model

**mmocr:**https://github.com/open-mmlab/mmocr

任务为Text Recognition