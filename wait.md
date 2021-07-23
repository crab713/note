collate_fn(nn.utils.DataLoader中、诡异bug中)







**诡异bug**:

mmseg.datasets.builder.py 145行 -> mmcv->parallel->collate.py batch=1



# new

### 1.编写验证部分代码

### 2.数据集

主要为compositon-1k

数据集介绍：
https://paperswithcode.com/dataset/composition-1k
https://paperswithcode.com/dataset/bg-20k
数据集获取：
https://mmediting.readthedocs.io/en/latest/matting_datasets.html
https://github.com/open-mmlab/mmediting/blob/master/docs/getting_started.md

### 3.反卷积方法



ocr

The Chars74K dataset

**mmocr:**https://github.com/open-mmlab/mmocr

任务为Text Recognition