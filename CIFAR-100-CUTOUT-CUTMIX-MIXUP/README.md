# CIFAR-100-CUTOUT-CUTMIX-MIXUP
### CIFAR-100数据集介绍：

author: 张飞21210980023

![image](https://user-images.githubusercontent.com/83007344/168309761-376b9fad-74b7-407d-b2b9-c4048a98521c.png)




使用Resnet-18算法训练测试CIFAR-100数据集，以及此数据集在CUTOUT、cutmix、mixup三种方下的准确度，并对训练测试结果进行对比。

### 其中的参数设置如下：

Batch_size:128

优化器: mini-batch momentum-SGD, 并采用L2正则化

loss function: Cross Entropy Loss

learning_rate: 如下所示：

Epoch    0-28    29-58   59-78   79-99

L_R      0.1     0.02    0.004   0.0008

### 运行方法：
在pycharm里运行cifrapytorch.py，可以在originnet3中获得log文件夹以及weights文件夹，运行visuallize.py可以获得cifar-100图片的可视化结果，结果保存在pic1文件夹。

### tensorbaord结果查看：
在pycharm的终端里输入tensorboard --logdir [项目地址]\CIFAR-100-CUTOUT-CUTMIX-MIXUP\originnet3\logs\baseline_test可以获得baseline的测试结果。其他结果类似。

 
### 参考资料
mixup:https://github.com/facebookresearch/mixup-cifar10
cutout:https://github.com/uoguelph-mlrg/Cutout
cutmix:https://github.com/clovaai/CutMix-PyTorch
