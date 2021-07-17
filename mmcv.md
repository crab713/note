基础库，分为两部分，一部分是pytorch的训练工具，一部分与dl无关的工具函数（IO/Image/Video）

## File IO

```python
import mmcv

data = mmcv.load('test.json')
data = mmcv.load('test.yaml')
data = mmcv.load('test.pkl')

# 从一个文件类别加载
with open('test.json', 'r') as f:
    data = mmcv.load(f)

# 将文件转储为字符串
json_str = mmcv.dump(data, file_format='json')

# 将数据转储为文件
mmcv.dump(data, 'out.pkl')

with open('test.yaml', 'w') as f:
    data = mmcv.dump(data, f, file_format='yaml')
```

## Image&video

采用opencv的方式实现

```python
mmcv.imread()
mmcv.imwrite()
mmcv.imshow()
# https://zhuanlan.zhihu.com/p/126725557
```

## util.config

```python
# 支持py,json,yaml
cfg = Config.fromfile('test.py')
assert cfg.a == 1
assert cfg.b.b1 == [0, 1, 2]
```



# 与pytorch有关的工具

