# 1.运行示例
前提：在`yaml`文件中已配置参数，模型文件已经定义。

**(1)运行有两种方式：**

1. 命令行运行：在终端输入命令 

```shell
python main.py -f testLMIB.yaml
```

2. pycharm编辑配置运行：形参设置为`-f testLMIB.yaml` 点击运行

**(2)可以通过命令行来修改从yaml文件中读取的参数**

```shell
python main.py -f testLMIB.yaml -r "{z_dim: 128, batch_size: 1024, nepochs: 300}" 
```
- **-r** 模型参数。yaml字符串，冒号之后一定要留空格，否则会出错。如：-r "{z_dim: 128, batch_size: 1024, nepochs: 300}"

更多具体参数，参考main.py 文件


# 2.示例设置参数
所需参数需在当前目录下的testLMIB.yaml文件中进行设置，格式为yaml，参数说明：
```
参数：    
      algorithm：算法的类名      
      data_path：数据集路径    
      metrics：验证指标列表        
      out_path:结果输出目录，默认为log目录
      over_write: 如果结果存在是否覆盖，默认是不覆盖
      algorithm_parameters:算法参数  
```
示例：
```yaml
data_path: data/handwritten.zip   # 指定数据存放位置，可以是zip文件或者目录
metrics: ['model.metric.ACC','model.metric.Pre']
over_write: True  # 是否覆盖已经运行过的结果。

#以下为模型和模型参数
algorithm: model.model.testLMIB_csv
algorithm_parameters:
    nepochs: 300
    z_dim: 128  # 此处根据需要设置任意参数。
```