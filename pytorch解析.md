# torch.autograd

## 包含内容

- orch.autograd.function （函数的反向传播）
- torch.autograd.functional （计算图的反向传播）
- torch.autograd.gradcheck （数值梯度检查）
- torch.autograd.anomaly_mode （在自动求导时检测错误产生路径）
- torch.autograd.grad_mode （设置是否需要梯度）
- model.eval() 与 torch.no_grad()
- torch.autograd.profiler （提供 function 级别的统计信息）



## function

例如nn.functional中的relu

网络的基本单元：**nn.Module**。

运算部分实现：**autograd functions**，内部定义了forward和backward用以描述前向和梯度反传的过程。

### 自定义autograd functions

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



## functional

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

## profiler

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



# BN & SyncBN

是对channel维度做归一化操作，性能与batch size有很大关系

## 作用

- 防止过拟合
- 加快收敛
- 防止梯度弥散

## pytorch实现

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



# Data

由Dataset、Sampler、DataLoader三部分组成，Dataset包装数据；Sampler决定采样方式；DataLoader负责总的调度，方便的在数据集上遍历。

## Dataset

Dataset 共有 Map-style datasets 和 Iterable-style datasets 两种：

### 1.1 Map-style dataset

`torch.utils.data.Dataset`

它是一种通过实现 `__getitem__()` 和 `__len()__` 来获取数据的 Dataset。

### 1.2 Iterable-style dataset

`torch.utils.data.IterableDataset`

它是一种实现 `__iter__()` 来获取数据的 Dataset。

## Sampler

`torch.utils.data.Sampler` 负责提供一种遍历数据集所有元素**索引**的方式。

对于所有的采样器来说，都需要继承Sampler类，必须实现的方法为`__iter__()`，当 DataLoader 需要计算len时需定义`__len__()`

PyTorch 也在此基础上提供了其他类型的 Sampler 子类

- `torch.utils.data.SequentialSampler` : 顺序采样样本，始终按照同一个顺序
- `torch.utils.data.RandomSampler`: 可指定有无放回地，进行随机采样样本元素
- `torch.utils.data.SubsetRandomSampler`: 无放回地按照给定的索引列表采样样本元素
- `torch.utils.data.WeightedRandomSampler`: 按照给定的概率来采样样本。样本元素来自 `[0,…,len(weights)-1]` ， 给定概率（权重）
- `torch.utils.data.BatchSampler`: 在一个batch中封装一个其他的采样器, 返回一个 batch 大小的 index 索引
- `torch.utils.data.DistributedSample`: 将数据加载限制为数据集子集的采样器。与 `torch.nn.parallel.DistributedDataParallel` 结合使用。 在这种情况下，每个进程都可以将 `DistributedSampler` 实例作为 `DataLoader` 采样器传递

## DataLoader

### 接口定义

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

