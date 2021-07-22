collate_fn(nn.utils.DataLoader中、诡异bug中)



[mmseg解析](https://blog.csdn.net/qq_32425195/article/details/110392397)

[mm系模型使用](https://zhuanlan.zhihu.com/p/163747610)

[mmlab官方解析](https://www.zhihu.com/people/openmmlab/)

[pytorch源码解读索引贴](https://zhuanlan.zhihu.com/p/328674159)





**诡异bug**:

mmseg.datasets.builder.py 145行 -> mmcv->parallel->collate.py batch=1



临时：

找Uper最终输出

验证

推理部分输出shape

说明文档





