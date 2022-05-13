# CIFAR-100-CUTOUT-CUTMIX-MIXUP
 对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现。

 其中的参数设置如下：

Batch_size:128
优化器: mini-batch momentum-SGD, 并采用L2正则化
loss function: Cross Entropy Loss
learning_rate:
    


 
参考资料
mixup:https://github.com/facebookresearch/mixup-cifar10
cutout:https://github.com/uoguelph-mlrg/Cutout
cutmix:https://github.com/clovaai/CutMix-PyTorch
