vscode的一些常用组件和操作



# 组件

1. **pylance**: 用于在vscode中获得联想关联跳转等一系列功能
2. **Remote-SSH**:用于通过SSH的方式，连接服务器
3. **jupyter**:用于在vscode中阅读该类文件



# 操作

1. 问题：在vscode中更改anaconda运行环境，方便代码提示

   解：快捷键ctrl+p，在弹出的框框中，输入 `>select interpreter` 来选择相应的Anaconda环境即可

2. 



## linux

查看文件

du -sh * 查看每个文件夹的大小

df -h 查看磁盘剩余空间

tar -zcvf data.tar.gz data 通过gzip的方式对文件进行打包压缩

rm -rf * 删库跑路

tar -xf data.tar.gz -c /data/ 解压到指定目录



### MobaXterm

一个用于SSH连接的软件