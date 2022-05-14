# 任务2(YOLO V3)

author: 唐寅21210980015

## 要求
在VOC数据集上训练并测试目标检测模型YOLO V3；

训练好后的模型可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并与Faster R-CNN对比，一共show六张图像；

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
VOC 07：使用VOC2007的train+val训练，然后使用VOC2007的test测试。

## 可视化三张VOC集外图像与YOLO V3进行对比

为了能更好地看出算法的性能，我们挑选了三张较难识别的图像，如下图所示(对于同一张图，前为Faster R-CNN检测结果，后为YOLO V3检测结果)：

<img src="https://user-images.githubusercontent.com/102893895/168336657-71cb6bc3-e923-44cf-90c9-f7972a9911b9.png"><img src="https://user-images.githubusercontent.com/102893895/168336743-f11a061d-7952-4432-9863-68c3a36b676e.png">

<img src="https://user-images.githubusercontent.com/102893895/168336845-18f33885-adf3-449e-89f7-9335f9ab4197.png"><img src="https://user-images.githubusercontent.com/102893895/168336870-f35be3a1-b08c-411c-9ddf-0aeadea3132f.png">

<img src="https://user-images.githubusercontent.com/102893895/168336957-4b7c20e8-687c-4de0-8d57-34ac0c8d66ed.png"><img src="https://user-images.githubusercontent.com/102893895/168336978-faf49e9d-83d8-45db-a606-3e42f92ca658.png">

由此可见，YOLO V3相较于Faster R-CNN对物体的检测更加灵敏，对图像中所占像素较少的物体有更强的捕捉能力。
