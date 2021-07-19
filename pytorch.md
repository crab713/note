## utils

```python
from torch.utils.checkpoint import checkpoint


# checkpoint函数需要输入requires_grad为True
# 不保留中间变量，再次求导
if self.use_checkpoint:
	x = checkpoint.checkpoint(blk, x)  # blk为层或函数，x为变量
else:
	x = blk(x)
```



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

