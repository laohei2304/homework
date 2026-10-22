                                                                      MNIST手写数字识别(PyTorch CNN)
基于PyTorch的MNIST手写数字分类实现。使用三层卷积层和两层全连接层，数据为CSV格式加载。在10个epoch的条件下，测试准确率约为100.000%

#1.项目结构
  homework/
    homework.1.py
    data/
      mnist_train.csv
      mnist_test.csv
      README.md

  #2.环境
    依赖             版本
    Python         3.12.5
    PyTorch        1.10
    pandas         1.3
    numpy          1.20

    bash
pip install torch pandas numpy

#模型结构
input 1 X 28 X 28
  Conv2d(1---32,3 X 3,pad=1）---ReLU---AvgPool(2)---32 X 14 X 14
  Conv2d(32---64,3 X 3,pad=1)---ReLU---AvgPool(2)---64 X 7 X 7
  Conv2d(64---128,3 X 3,pad=1)---Relu---AvgPool(2)---128 X 3 X 3
  Flatten
  Linear(1152---512)---ReLU
  Linear(512---10）
#训练过程
  Adam(lr=0.001),batch=64,共10个epoch。像素先归一化并reshape为（N，1,28,28），训练集打乱。
  
  Epoch         Loss
   1           1.28138
   2           0.45400
   3           0.27637
   4           0.20734
   5           0.15207
   6           0.13126
   7           0.10061
   8           0.09104
   9           0.08220
   10          0.07222

#感悟
  在此次项目中，我感悟颇深，学习到以下几点
    1.linear为全连接网络，cnn为卷积神经网络，在卷积神经网络中会用到linear，而两者主要区别在于有无卷积层
    2.对cnn的理解更加深入，一张图片会经过卷积，池化，再卷积，再池化的过程
    3.在训练过程中 ，若准确率过低，loss没有递减，那么需要修复csv文件
    4.csv和Dataset不能混用，两种加载模式完全不同
    5.lr是学习率
    6.平均池化可以避免过拟合，而最大池化在接受信号方面更好
    7.学习次数越高，准确率越高，但次数过高会出现过拟合
    8.卷积次数越多，拟合效果越好，但过高同样出现过拟合
