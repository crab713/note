用于为一个系统构建config文件

[知乎解析](https://zhuanlan.zhihu.com/p/366289700)

## 使用

需要创建`CN()`这个作为容器来装载我们的参数，这个容器可以嵌套

```python
from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'test'
_C.model = CN()  # 嵌套使用
_C.model.backbone = 'resnet'
_C.model.depth = 18

print(_C)  
'''
  name: test
  model:
      backbone: resnet
      depth: 18
'''

```

## API

#### 1.merge_from_file()

比较默认参数与特定参数不同的部分，用特定参数覆盖，不能有默认参数中不存在的参数

```python
_c.merge_from_file("./config.yaml")
```



#### 2.merge_from_list()

从list中读取，一个参数名跟一个参数值，同样需要默认参数中存在以下参数

```python
opts = ["name", 'test_name', "model.backbone", "vgg"]
__C.merge_from_list(opts)
'''
model:
  backbone: vgg
name: test_name
'''
```

#### 3.freeze()

冻结参数，不允许进行修改

```python
_C.freeze()
```



#### 4.defrost()

解除freeze()，修改参数

