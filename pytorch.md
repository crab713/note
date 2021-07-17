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





## Dataparallel & distributed

