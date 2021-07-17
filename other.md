### BN

syncBN:在多张卡上同步BN，避免在单卡上bz=1，BN无法发挥作用的问题

[pytorch的坑](https://zhuanlan.zhihu.com/p/59271905)



### warmup

使用较低的learning rate预热，慢慢恢复到正常

[mmlab宝贝网站](https://openmmlab.com/)



### 模型训练次数指标

iter--一次迭代，指一个min_batch的一次forward+backward

epoch--指迭代完一次所有数据

