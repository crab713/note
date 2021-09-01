# 用途

用于处理HDF5文件(即.h5文件)，HDF文件是安装树状结构组织起来的。其顶部是根节点（），根节点下可以接很多分组（group），每个分组下有可以有很多节点，包括表（table），数组（array），压缩数组（compression array，Earray），可扩展压缩数组（enlargeable array，Earray），变长数组（variable length array，VLarray）。每个节点下还有叶子节点，即最终存储的数据，该文件用于处理大规模数据时使用



# 用法

### 导入

``` python
import tables
from tables import *
```



### 读取.h5文件

示例文件test.h5下有sentences(earray)及indices(table)

``` python
# 打开文件
table = tables.open_file('test.h5', driver="H5FD_CORE")

# 获得分支数据
contexts = table.get_node('/sentences')[:].astype(np.long)
indexs = table.get_node('/indices')[:]

index = indexs[0]
pos_utt, ctx_len, res_len = index['pos_utt'], index['ctx_len'], index['res_len']
```



### 写入earray

``` python
table = tables.open_file(save_dir, mode='w')

filters = tables.Filters(complevel=5, complib='blosc')
sentences = np.array([1,2,3,4,5])
earray = table.create_earray(
    table.root,
    'sentences',
    tables.Atom.from_dtype(sentences.dtype),
    shape=[0],
    filters=filters,
)
# 写入的数据均需要为nparray格式
earray.append(sentences)
```



### 写入table

``` python
# 构建表
class particle(IsDescription):
    ctx_len = Int32Col()
    pos_utt = Int32Col()
    res_len = Int32Col()

indices = np.array([[1,2,3],[3,4,5]])
indice_table = table.create_table(
    table.root,
    'indices',
    particle,
    chunkshape=3
)
indice_table.append(indices)
```



