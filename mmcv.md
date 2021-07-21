开源框架，分为两部分，一部分是pytorch的训练工具，一部分与dl无关的工具函数（IO/Image/Video）。从组件上分，常用组件有**fileio、image、parallel、runner、utils、config、Hook**

## File IO

该组件实现基础为**file_handlers**,再对外读写接口**load**，屏蔽掉具体子类handler

#### 1. 使用

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

#### 2. 自定义扩展

```python
# 注册一个.npy文件读取
@register_handler('npy') # 将npy注册进file_handlers
class NpyHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        return np.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        np.save(file, obj)

    def dump_to_str(self, obj, **kwargs):
        # 实际上这么写没有意义，这里只是举例
        return obj.tobytes()
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

#### 1. 使用

可通过参数use_predefined_variables实现自动替换预定义变量功能

```python
# 支持py,json,yaml,dict
cfg = Config.fromfile('test.py')
```



#### 2. 合并多个配置文件

从多个base文件中合并，在非base的最终配置文件中添加_base_字段

如果多个配置文件中有相同的参数名，则参数值选用主配置文件中的

```python
# 主配置文件中
_base_=['./base.py']

```

#### 3. 从字典中合并

``` python
input_options = {'item2.a': 1, 'item2.b': 0.1, 'item3': False}

_delete_=True # 忽略base相关配置，直接采用新配置字段
# 同名字段以新合并进的为主
cfg.merge_from_dict(input_options)
```



## Registry

提供全局类注册器功能，用完全相似的对外修饰函数来管理构建不同组件，内部维护的是一个全局key-value对。

``` python
# 0. 先构建一个全局的 BACKBONE 注册器类
BACKBONES = mmcv.utils.Registry('backbone')

# 1. 将具体类传入注册器中，可以不传入参数，此时默认实例化的配置字符串是 str (类名)
@BACKBONES.register_module(name='SwinTransformer')
class SwinTransformer(nn.Module)
BACKBONE.register_module(SwinTransformer) # 两种方式

# 类实例化
CATS.get('SwinTransformer')(**args)
```



## Hook

有多种默认Hook，Checkpoint，LrUpdater，optimizer，TextLogger，IterTimer等

可以通过hook名_config来配置对应的参数

``` python
checkpoint_config = dict(by_epoch=False, interval=16000)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
```

#### 1.CheckpointHook

通过**max_keep_ckpts**参数设置最多保存多少个权重文件,**sync_buffer**参数为True则在保存前对模型进行跨卡同步

以epoch为单位，实现after_train_epoch方法即可，否则仅实现 after_train_iter 方法即可

#### 2.LrUpdaterHook

TODO



## Runner

根源为mmcv.runner.base_runner.py->BaseRunner:class

是训练部分的引擎，使用过程分为4步骤：

1. Runner对象初始化
2. 注册各类Hook到Runner
3. 调用Runner的resume或者load_checkpoint方法对权重重载
4. 运行给定工作流，开始训练

#### 1. Runner初始化

```python
# 抽象基类
class BaseRunner():
def __init__(self,
             model,
             batch_processor=None, # 已废弃
             optimizer=None,
             work_dir=None,
             logger=None,
             meta=None, # 提供了该参数，则会保存到 ckpt 中
             max_iters=None, # 这两个参数非常关键，如果没有给定，则内部自己计算
             max_epochs=None):
```



#### 2. 注册Hook

**register_training_hooks**,作用是注册一些默认Hook，如下：

```python
def register_training_hooks(self,
                            lr_config, # lr相关
                            optimizer_config=None, # 优化器相关
                            checkpoint_config=None, # ckpt 保存相关
                            log_config=None, # 日志记录相关
                            momentum_config=None, # momentum 相关
                            timer_config=dict(type='IterTimerHook')) # 迭代时间统计
```



**register_hook**，除了上面的Hook之外，通过该方式注入，如eval_Hook，将Hook类实例插入_hooks列表中

``` python
def register_hook(self, hook, priority='NORMAL')
```



#### 3. resume或load_checkpoint

resume 方法用于训练过程中停止然后恢复训练时加载权重，而 load_checkpoint 仅仅是加载预训练权重而已



#### 4. run

``` python
# EpochBasedRunner.run()
def run(self, 
    data_loaders, # dataloader 列表
    workflow,  # 工作流列表，长度需要和 data_loaders 一致
    max_epochs=None, 
    **kwargs)
```

假设只想运行训练工作流，则可以设置 workflow = [('train', 1)]，表示 data_loader 中的数据进行迭代训练

workflow = [('train', 3), ('val',1)]，表示先训练 3 个 epoch ，然后切换到 val 工作流,如果有两个工作流，则data_loaders中也需要两个数据集

``` python
# train中的训练部分
for i, data_batch in enumerate(self.data_loader):
    self.call_hook('before_train_iter')
    self.run_iter(data_batch, train_mode=True)
    self.call_hook('after_train_iter')
```

#### 5. 原理解析

1. 模型的组装位于models.segmentors.encoder_decoder.py->init()中，通过传入cfg字典来获得配置值，构建backbone等组件，数据流通为在backbone中保存各层的输出outputs，return回模型中供decoder等使用
runner继承于base基类，得到iter和epoch两种迭代类型的runner，以iter_runner为例，实际train.py中调用的为该类中的run方法

2. 在run方法内，需要根据当前工作流所处状态，选择是train或val这两种工作流，通过getattr方法获得对应的函数方法iter_runner=train()/val()

   通过运行该方法，传入该iter所需的数据（大小batch_size）开始对模型的一次训练

3. 以train()为例，模型训练的关键代码为`output = self.model.train_step(data_batch, self.optimizer, **kwargs)`

   这里的model为train.py中传入runner的模型，对应于mmseg.models.segmentors中的类，也就要求必须在模型类中必须要有train_step()函数供调用，val_step()同理

4. 对于encoder_decoder类，需要实现forward_train()和forward_test()两个方法，供forward方法调用(base基类中)

   ``` python
   def forward(...):
       if return_loss:
           return self.forward_train(...)
       else:
           return self.forward_test(...)
   ```

5. 回到runner中，更新学习率部分代码在OptimizerHook类中after_train_iter方法，更新所使用的loss值为decode.loss与aux.loss相加的值，相加部分代码在segmentors的base基类_parse_losses中实现

6. 总结：关键源代码位置如下

   * mmseg.apis.train.py
   * mmseg.models.segmentors.base.py/encoder_decoder.py
   * mmcv.runner.base_runner.py/iter_base_runner.py
   * mmcv.runner.hooks.optimizer.py->after_train_iter(self,runner)

