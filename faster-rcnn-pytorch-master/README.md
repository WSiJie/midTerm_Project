# 任务2(Faster R-CNN)
## 要求
在VOC数据集上训练并测试目标检测模型Faster R-CNN；

在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；

训练好后的模型可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并与YOLO V3对比，一共show六张图像；

## 数据集介绍(VOC)
PASCAL VOC挑战赛(The PASCAL Visual Object Classes)是一个世界级的计算机视觉挑战赛,PASCAL全称：Pattern Analysis, Statical Modeling and Computational Learning，是一个由欧盟资助的网络组织。

很多优秀的计算机视觉模型比如分类，定位，检测，分割，动作识别等模型都是基于PASCAL VOC挑战赛及其数据集上推出的。

PASCAL VOC从2005年开始举办挑战赛，每年的内容都有所不同，从最开始的分类，到后面逐渐增加检测，分割，人体布局，动作识别(Object Classification 、Object Detection、Object Segmentation、Human Layout、Action Classification)等内容，数据集的容量以及种类也在不断的增加和改善。该项挑战赛催生出了一大批优秀的计算机视觉模型(尤其是以深度学习技术为主的)。这项挑战赛已于2012年停止举办了，但是研究者仍然可以在其服务器上提交预测结果以评估模型的性能。

对于现在的研究者来说比较重要的两个年份的数据集是 PASCAL VOC 2007 与 PASCAL VOC 2012，这两个数据集频频在现在的一些检测或分割类的论文当中出现。

PASCAL VOC 数据集的20个类别及其层级结构：

<div align=center><img width="475" alt="截图" src="https://user-images.githubusercontent.com/102893895/168310116-74225eef-60ea-438d-90df-8db0f5aef6df.png"></div>

+ 从2007年开始，PASCAL VOC每年的数据集都是这个层级结构
+ 总共四个大类：vehicle,household,animal,person
+ 总共20个小类，预测的时候是只输出图中黑色粗体的类别
+ 数据集主要关注分类和检测，也就是分类和检测用到的数据集相对规模较大。关于其他任务比如分割，动作识别等，其数据集一般是分类和检测数据集的子集

## 训练集测试集划分
VOC 07+12：使用VOC2007的train+val和VOC2012的train+val训练，然后使用VOC2007的test测试。

## Faster R-CNN网络结构
如图所示

<div align=center><img width="305" alt="截图" src="https://user-images.githubusercontent.com/102893895/168313121-3be632b9-2a0f-486e-82c1-e0d07a8e3a86.png"></div>

依据原作者的观点，Faster R-CNN主要可以分为4个部分：

+ Conv layers：作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层
+ Region Proposal Networks：RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals
+ Roi Pooling：该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别
+ Classification：利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置

## 实验设置
模型会预先加载已经训练好的backbone网络ResNet50，训练过程分为“冻结阶段”与“解冻阶段”

+ batch size：冻结阶段batch size为12，解冻阶段batch size为6
+ learning：1/10000，cos学习率下降策略，最低下降为初始学习率的1/100
+ 优化器：adam
+ epoch：冻结阶段50，解冻阶段50
+ loss function：总损失分为四个部分：
  + rpn_loc_loss：一阶段预测框回归误差，smooth L1 loss
  + rpn_cls_loss：一阶段预测框分类误差(判断是否为物体)，cross-entropy loss
  + roi_loc_loss：二阶段预测框回归误差，smooth L1 loss
  + roi_cls_loss：二阶段预测框分类误差(判断是否为某一具体类别)，cross-entropy loss
+ 评价指标：mAP

## 模型训练
**1、处理标签**

进入`voc_annotation.py`修改

```python
annotation_mode =2
```

**2、训练模型**

运行`train.py`

运行之前，会自动加载backbone网络，这里代码部分默认加载ResNet50网络(主干网络需要存于model_data目录中)

```python
model_path = 'model_data/voc_weights_resnet.pth'
```

```python
backbone = "resnet50"
```

相关主干权重网络百度网盘链接如下：
+ [backbone网络权重文件](https://pan.baidu.com/s/1CPywMMEv1xkXj6wU78GZKQ?pwd=0001)
+ 密码：0001

**3、Loss曲线**

利用tensorboard可视化后的测试集和验证集Loss曲线如下：

![loss曲线](https://user-images.githubusercontent.com/102893895/168323524-53163743-c982-4e65-b693-e8be44d8cc80.png)

![val_loss曲线](https://user-images.githubusercontent.com/102893895/168323577-4a2a191c-def5-4918-9770-a216e5256e07.png)

在同一坐标下展示如下：

<div align=center><img width="900", img height="500", src=https://user-images.githubusercontent.com/102893895/168324776-a5f15ace-c26b-4d9f-ad4d-f7b9bfb64b6c.png></div>



## 模型测试











保存模型百度网盘链接

## Reference
