## utils

```python
from torch.utils.checkpoint import checkpoint


# checkpoint函数需要输入requires_grad为True
# 不保留中间变量，再次求导
if self.use_checkpoint:
	x = checkpoint.checkpoint(blk, x)  # blk为层或函数，x为变量
else:
	x = blk(x)
    
    
# 2d反卷积 只恢复尺寸，不恢复数值 https://www.zhihu.com/question/48279880
class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True)


# 生成等差数列（可用于drop_path_rate的参数）
import torch.linspace
a = torch.linspace(0,10,steps=5) # 0-10之间5个数构成的等差数列

# 做填充层，方便后续训练进行更换，分步训练
nn.Identity()

```



## 维度交换与变换

在torch库中，有transpose和permute两种方式进行维度交换

``` python
x = x.transpose(1, 2) # 仅能两个维度之间交换
x = x.permute(0,3,1,2) # 多维度进行交换
```

维度变换主要有**reshape**和**view**方法,他们的区别在于reshape可作用于非连续的张量上，而view只能作用于连续的张量上

``` python
x = torch.tensor(2,3,3)

x = x.reshape(6,3)
x = x.view(6,3)
```



## 维度合并与切分

主要有**cat**和**split**两个方法，cat为对向量进行合并，split为切分

``` python
# 按dim维对向量进行拼接
c = torch.cat((a,b), dim=0) 

# 按dim维对向量进行切分，以及切分的大小split_size
# torch.split(tensor, split_size_or_sections, dim=0)
y = torch.split(x,3,dim=0)#按照4这个维度去分，每大块包含3个小块
for i in y:
    print(i.size())
 
output:
torch.Size([3, 8, 6])
torch.Size([3, 8, 6])
torch.Size([1, 8, 6])
```



## 张量复制

#### 1.clone()

clone()函数返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯，对a_进行的运算梯度会加在a的梯度上。

``` python
a = torch.tensor(1.0, requires_grad=True)
a_ = a.clone()
```

#### 2.detach()

detach()函数返回一个和源张量同shape、dtype和device的张量，并且与源张量共享数据内存，但不提供梯度的回溯

``` python
a_ = a.detach()
```

#### 3. new_tensor()

new_tensor()可以将源张量中的数据复制到目标张量（数据不共享），同时提供了更细致的属性控制

``` python
f = a.new_tensor(a, device="cpu", dtype=torch.float64, requires_grad=False)
```



## Tensor数值操作

#### 1. where

`torch.where(condition, x, y) → Tensor`

此方法是将x中的元素和条件相比，如果符合条件就还等于原来的元素，如果不满足条件的话，那就去y中对应的值,xy均为tensor

``` python
torch.where(x > 0, x, y)
```

#### 2. sum

`torch.sum(input, list: dim, bool: keepdim=False, dtype=None) → Tensor`

**dim**:要求和的维度，根据该维度求和，返回的值为一个tensor数组，为none时返回全部数的和



## datach() && Variable()

#### Variable()

Variable是对Tensor的一个封装，操作与Tensor相同，但是每个Variable都有三个属性，Varibale的Tensor本身的.data，对应Tensor的梯度.grad，以及这个Variable是通过什么方式得到的.grad_fn

在模型训练过程中，会建立一张计算图，梯度计算会在这张图内，而在训练完成后新建的Tensor并不在该图内，无法用该Tensor计算loss，因此需要使用Variable()

``` python

from torch.autograd import Variable
import torch

y_tensor = torch.randn(10,5)
x = Variable(x_tensor,requires_grad=True) #Varibale 默认时不要求梯度的，如果要求梯度，需要说明
```

#### detach()

神经网络的训练有时候可能希望保持一部分的网络参数不变，只对其中一部分的参数进行调整；或者值训练部分分支网络，并不让其梯度对主网络的梯度造成影响,该方法用来切断一些分支的反向传播

**torch.tensor.detach()**用法介绍：

（1）返回一个新的从当前图中分离的Variable。

（2）返回的 Variable 不会梯度更新。

（3）被detach 的Variable volatile=True， detach出来的volatile也为True。

（4）返回的Variable和被detach的Variable指向同一个tensor。



## 上采样

`CLASS torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)`

**参数：**

1. **size**:指定输出大小
2. **scale_factor**:指定输出为输入的多少倍，与size只能填一个
3. **mode**:可使用的上采样算法，有`'nearest'`, `'linear'`, `'bilinear'`, `'bicubic'` and `'trilinear'`.默认使用'nearest'`
4. **align_corners**：如果为True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。仅当使用的算法为`'linear'`, `'bilinear'`or `'trilinear'时可以使用。``默认设置为``False`



## distributedDataParallel

相较于Dataparallel，distributed具有很大优势[解析](https://blog.csdn.net/weixin_41041772/article/details/109820870)



```python
torch.distributed.is_available() # 判断分布式包是否可得

# 分布式进程组初始化 backend=['mpi','gloo','nccl']
torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name='')

torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)


# 不同进程间的数据同步，通过阻塞栅栏
torch.distributed.barrier()
```



## 固定随机值

参数默认是进行随机初始化，使训练时初始化时固定，能够复现结果

```python
torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的

torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子；
```



## 替换layer

```python
self.add_module(name, layer) # 添加层并赋予name，同name的话会覆盖
getattr(self, name) # 获取这层



# 使用model.buffers()查看网络基本结构
<bound method Module.buffers of ResNet(
 ...
 (fc): Linear(in_features=512, out_features=1000, bias=True)
)

# 将fc层进行替换
fc_in_features = model.fc.in_features # 获取输入大小
model.fc = torch.nn.Linear(fc_in_feature, 2, bias=True)
```







# 基础构建部分



### 构建数据集dataset

继承torch的dataset

需要重写__getitem__和__len__方法，并在__getitem__中将数据预处理成tensor格式

```python
    
from torch.utils.data import Dataset
import torch
from sklearn.utils import shuffle
from config.NerConfig import NerConfig

class CLSDataset(Dataset):
    def __init__(self, corpus_path, word2idx, max_seq_len, data_regularization=False):
    self.data_regularization = data_regularization
    self.nerConfig = NerConfig# define special symbols
    self.pad_index = 0
    self.unk_index = 1

    # 加载语料 train_text=[size,[sentence,label]]
    self.train_text = []
    with open(corpus_path, "r", encoding="UTF-8") as f:
        
# 重写__len__方法
def __len__(self):
    return self.train_line

# 重写并对数据预处理，以torch.tensor的形式输出
def __getitem__(self, item):
    output = {"text_input": torch.tensor(text_input, dtype=torch.long),"label": torch.tensor(label_input)}
    return output
```



### 构建模型

继承nn.Module

如果使用GPU训练，则需要将模型以及模型内生成的tensor都传入cuda中，使用to(device)方法

需要重写forward方法

```python
class BiLSTM_CRF(nn.Module):  
def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, device="cpu"):
    super(BiLSTM_CRF, self).__init__()
    self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
    self.word_embeds.to(self.device)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                        num_layers=1, bidirectional=True)
    self.lstm.to(self.device)

    # Maps the output of the LSTM into tag space.
    self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
    self.hidden2tag.to(self.device)
    
    def forward(self, sentence):
    # Get the emission scores from the BiLSTM
    lstm_feats = self._get_lstm_features(sentence)

    # Find the best path, given the features.
    score, tag_seq = self._viterbi_decode(lstm_feats)
    return score, tag_seq
```


### 模型训练

##### 1. 载入训练集

```python
self.train_dataset = CLSDataset(corpus_path=self.config["train_corpus_path"],
                                word2idx=self.word2idx,
                                max_seq_len=max_seq_len,
                                data_regularization=False)
self.train_dataloader = DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                num_workers=0,
                                shuffle=True,
                                collate_fn=lambda x: x) # 动态padding到同样长度
```



##### 2.判断是否有可用GPU

```python
cuda_condition = torch.cuda.is_available() and with_cuda
self.device = torch.device("cuda:0" if cuda_condition else "cpu")

self.ner_model = BiLSTM_CRF(device=self.device)
self.ner_model.to(self.device)
```



##### 3.保存与载入模型

```python
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(self.device)

# 以GPU下的参数保存
model.to("cuda")
torch.save({"model_state_dict": model.state_dict()}, save_path)
model.to(self.device)
```



##### 4. 多分类f1_score计算

```python
score = f1_score(all_tags, all_predict, average='weighted') # weighted标识会计算不同分类的权重
```



##### 5.tqdm进度条使用

```python
import tqdm

data_iter = tqdm.tqdm(enumerate(data_loader),
                      desc="EP_%s:%d" % (str_code, epoch),
                      total=len(data_loader),
                      bar_format="{l_bar}{r_bar}")
for i, data in data_iter:
    self.ner_model.zero_grad()

    loss = self.ner_model.neg_log_likelihood(sentence, tags)
    total_loss += loss.item()
    log_dic = {
        "epoch": epoch,
        "train_loss": loss.item()/(i+1),
    }
    data_iter.write(str({k: v for k, v in log_dic.items()})) # 打印当前状态
    
    loss.backward()
    self.optimizer.step()
```

