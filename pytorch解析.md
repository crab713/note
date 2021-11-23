# 1 torch.autograd

## 1.1 包含内容

- orch.autograd.function （函数的反向传播）
- torch.autograd.functional （计算图的反向传播）
- torch.autograd.gradcheck （数值梯度检查）
- torch.autograd.anomaly_mode （在自动求导时检测错误产生路径）
- torch.autograd.grad_mode （设置是否需要梯度）
- model.eval() 与 torch.no_grad()
- torch.autograd.profiler （提供 function 级别的统计信息）



## 1.2 function

例如nn.functional中的relu

网络的基本单元：**nn.Module**。

运算部分实现：**autograd functions**，内部定义了forward和backward用以描述前向和梯度反传的过程。

### 1.2.1 自定义autograd functions

``` python
class Exp(Function):                    # 此层计算e^x
    @staticmethod
    def forward(ctx, i):                # 模型前向
        result = i.exp()
        ctx.save_for_backward(result)   # 保存所需内容，以备backward时使用，所需的结果会被保存在saved_tensors元组中；此处仅能保存tensor类型变量，若其余类型变量（Int等），可直接赋予ctx作为成员变量，也可以达到保存效果
        return result
    @staticmethod
    def backward(ctx, grad_output):     # 模型梯度反传
        result, = ctx.saved_tensors     # 取出forward中保存的result
        return grad_output * result     # 计算梯度并返回

a = torch.tensor([1.], requires_grad=True)
result = Exp.apply(a)  // forward
result.backward()
print(a.grad)
```



## 1.3 functional

1. 计算图的反向传播`loss.backward()`，实际是调用`autograd.backward()`接口

2. 会根据用户操作建立有向无环图，非叶节点通过`grad_fn()`函数实现梯度反传，叶节点不具有`grad_fn()`函数，但有梯度累积函数

3. `grad_fn()`由autograd_function中tensor的操作决定

4. 中间节点的梯度不会保存，节约空间

5. 计算图在每次前向传播(**forward**)时都是从头构建

6. 需存在节点requires_grad=True才会构建计算图，否则报错

7. 可对tensor注册hook，在反向传播时调用，常用于中间节点

   ```python
   def variable_hook(grad):                       
       print('the gradient of C is：', grad)
       
   D = C.exp()
   D.register_hook(variable_hook)
   ```

8. `torch.no_grad()`关闭自动求导，不计算梯度，节省空间

   `model.eval()`使Dropout等采用eval mode，更准确，两者实际没有关联。

## 1.4 profiler

`torch.autograd.profiler` （提供function级别的统计信息）

`profiler.profile()`：得到每个kernel的cpu耗时信息

```python
import torch
from torchvision.models import resnet18

x = torch.randn((1, 3, 224, 224), requires_grad=True)
model = resnet18()
with torch.autograd.profiler.profile() as prof:
    for _ in range(100):
        y = model(x)
        y = torch.sum(y)
        y.backward()
# NOTE: some columns were removed for brevity
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```



# 2 BN & SyncBN

是对channel维度做归一化操作，性能与batch size有很大关系

## 2.1 作用

- 防止过拟合
- 加快收敛
- 防止梯度弥散

## 2.2 pytorch实现

相关类在`torch.nn.modules.batchnorm`中，如下：

- `_NormBase`：`nn.Module` 的子类，定义了 BN 中的一系列属性与初始化、读数据的方法；
- `_BatchNorm`：`_NormBase` 的子类，定义了 `forward` 方法；
- `BatchNorm1d` & `BatchNorm2d` & `BatchNorm3d`：`_BatchNorm`的子类，定义了不同的`_check_input_dim`方法(验证输入合法性)。

``` python
def dummy_bn_forward(x, bn_weight, bn_bias, eps, mean_val=None, var_val=None):
    if mean_val is None:
        mean_val = x.mean([0, 2, 3])
    if var_val is None:
        # 这里需要注意，torch.var 默认算无偏估计，因此需要手动设置unbiased=False
        var_val = x.var([0, 2, 3], unbiased=False)

    x = x - mean_val[None, ..., None, None] # 对channel维操作，mean_val.shape=[channel]
    x = x / torch.sqrt(var_val[None, ..., None, None] + eps)
    x = x * bn_weight[..., None, None] + bn_bias[..., None, None]
    return mean_val, var_val, x
```



# 3 Data

由Dataset、Sampler、DataLoader三部分组成，Dataset包装数据；Sampler决定采样方式；DataLoader负责总的调度，方便的在数据集上遍历。

## 3.1 Dataset

Dataset 共有 Map-style datasets 和 Iterable-style datasets 两种：

### 3.1.1 Map-style dataset

`torch.utils.data.Dataset`

它是一种通过实现 `__getitem__()` 和 `__len()__` 来获取数据的 Dataset。

### 3.1.2 Iterable-style dataset

`torch.utils.data.IterableDataset`

它是一种实现 `__iter__()` 来获取数据的 Dataset。

## 3.2 Sampler

`torch.utils.data.Sampler` 负责提供一种遍历数据集所有元素**索引**的方式。

对于所有的采样器来说，都需要继承Sampler类，必须实现的方法为`__iter__()`，当 DataLoader 需要计算len时需定义`__len__()`

PyTorch 也在此基础上提供了其他类型的 Sampler 子类

- `torch.utils.data.SequentialSampler` : 顺序采样样本，始终按照同一个顺序
- `torch.utils.data.RandomSampler`: 可指定有无放回地，进行随机采样样本元素
- `torch.utils.data.SubsetRandomSampler`: 无放回地按照给定的索引列表采样样本元素
- `torch.utils.data.WeightedRandomSampler`: 按照给定的概率来采样样本。样本元素来自 `[0,…,len(weights)-1]` ， 给定概率（权重）
- `torch.utils.data.BatchSampler`: 在一个batch中封装一个其他的采样器, 返回一个 batch 大小的 index 索引
- `torch.utils.data.DistributedSample`: 将数据加载限制为数据集子集的采样器。与 `torch.nn.parallel.DistributedDataParallel` 结合使用。 在这种情况下，每个进程都可以将 `DistributedSampler` 实例作为 `DataLoader` 采样器传递

## 3.3 DataLoader

### 3.3.1 接口定义

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

### 3.3.2 自动批处理

在使用 sampler 产生的 indices 获取采样到的数据时，DataLoader 使用 `collate_fn` 参数将样本列表整理成 batch。抽象这个过程，其表示方式大致如下

```python
# For Map-style
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])

# For Iterable-style
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```

### 3.3.3 代码详解

for 循环会调用 dataloader 的 `__iter__(self)` 方法(根据num_workers)，以此获得迭代器来遍历 dataset

### 3.3.4 注意

1. 不建议在多进程加载中返回CUDA张量

2. 每一个DataLoader都包含迭代器

3. **整个过程调用关系总结** 如下：

   `loader.__iter__` --> `self._get_iterator()` --> `class _SingleProcessDataLoaderIter` --> `class _BaseDataLoaderIter` --> `__next__()` --> `self._next_data()` --> `self._next_index()` -->`next(self._sampler_iter)` 即 `next(iter(self._index_sampler))` --> 获得 index --> `self._dataset_fetcher.fetch(index)` --> 获得 data



# 4 nn.Module

## 4.1 特点

- 一般有一个基类来定义接口，通过继承来处理不同维度的 input
- 每一个类都会有一个对应的nn.functional函数，类定义所需要的arguments和模块的parameters，在forward中传给nn.functional的对应函数来实现
- 继承nn.Module的模块主要重载**init**、**forward**、extra_repr、reset_parameters函数

## 4.2 状态转换

- 训练&&测试：`self.train(), self.eval()`，会通过**self.children()**调整所有子模块。

- freeze：重载module的**train**函数

  ``` python
  def train(self, mode=True):
      super(ResNet, self).train(mode)
      self._freeze_stages()
      if mode and self.norm_eval:
          for m in self.modules():
              # trick: eval have effect on BatchNorm only
              if isinstance(m, _BatchNorm):
                  m.eval()
  ```

  

- 梯度的处理：通过**requires_grad_** 和 **zero_grad** 函数，他们都调用了 **self.parameters()** 来访问所有的参数

  ``` python
  def requires_grad_(self: T, requires_grad: bool = True) -> T:
      for p in self.parameters():
          p.requires_grad_(requires_grad)
      return self
  ```

## 4.3 参数转换

nn.Module 实现了如下 8 个常用函数将模块转变成 float16 等类型、转移到 CPU/ GPU上。

1. **CPU**：将所有 parameters 和 buffer 转移到 CPU 上
2. **type**：将所有 parameters 和 buffer 转变成另一个类型
3. **CUDA**：将所有 parameters 和 buffer 转移到 GPU 上
4. **float**：将所有浮点类型的 parameters 和 buffer 转变成 float32 类型
5. **double**：将所有浮点类型的 parameters 和 buffer 转变成 double 类型
6. **half**：将所有浮点类型的 parameters 和 buffer 转变成 float16 类型
7. **bfloat16**：将所有浮点类型的 parameters 和 buffer 转变成 bfloat16 类型
8. **to**：移动模块或/和改变模块的类型

这些函数的功能最终都是通过 `self._apply(function)` 来实现的， function 一般是 lambda 表达式或其他自定义函数。

## 4.4 Apply函数

_apply() 是**专门针对 parameter 和 buffer** 而实现的一个“仅供内部使用”的接口，但是 apply 函数是“公有”接口

### 源码

``` python
def apply(self: T, fn: Callable[['Module'], None]) -> T:
    for module in self.children():
        module.apply(fn)
    fn(self)
    return self
```

### 样例

``` python
@torch.no_grad()
def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        m.weight.fill_(1.0)
        print(m.weight)

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)
```

## 4.5 增删改查

nn.module类中的几种参数类型：parameters、buffer、module：attribute

基类中通过重写`__delattr__` ， `__setattr__`，`__getattr__`从而实现管理

### 4.5.1 属性修改

对 nn.Module 属性的修改有一下三个函数，函数以及对应功能如下

1. add_module：增加子神经网络模块，更新 self._modules
2. register_parameter：增加通过 BP 可以更新的 parameters 
3. register_buffer：增加不通过 BP 更新的 buffer

在日常的代码开发过程中，更常见的用法是直接通过`self.xxx = xxx` 的方式来增加或修改，本质上会调用 nn.Module 重载的函数 `__setattr__`

**注意**：**self.xxxx = torch.Tensor() 是一种不被推荐的行为**，因为这样新增的 attribute 既不属于self.parameters，也不属于 self.buffers，而会被视为普通的 attribute，模块状态转换时会出现bug

### 4.5.2 属性删除

属性的删除通过重载的 `__delattr__` 来实现，会挨个检查 self._parameters、self._buffers、self._modules 和普通的 attribute 并将 name 从中删除。

``` python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.register_buffer('test', torch.tensor([1,2,3]))
        self.__delattr__('test')
```

### 4.5.3 属性访问

nn.Module中提供了常用函数，都会返回一个迭代器用于访问模块中的buffer，parameter，子模块等。

1. parameters：调用 self.named_parameters 并返回模型参数，被应用于 self.requires_grad_ 和 self.zero_grad 函数中
2. named_parameters：返回 self._parameters 中的 name 和 parameter 元组，如果 recurse=True 还会返回子模块中的模型参数
3. buffers：调用 self.named_buffers 并返回模型参数
4. named_buffers：返回 self._buffers 中的 name 和 buffer 元组，如果 recurse=True 还会返回子模块中的模型 buffer
5. children：调用 self.named_children，只返回 self._modules 中的模块，被应用于 self.train 函数中
6. named_children：只返回 self._modules 中的 name 和 module 元组
7. modules：调用 self.named_modules 并返回各个 module 但不返回 name
8. named_modules：返回 self._modules 下的 name 和 module 元组，并递归调用和返回 module.named_modules
9. **module.attribute**：等价于`getattr(module, 'attribute')`

