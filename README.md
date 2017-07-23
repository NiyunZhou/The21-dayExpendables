# 思路及模型简述
数据集分为Frame level data和Video level data。youtube上的视频经过筛选之后，最多取前300s，每秒取一帧，一共得到最多300帧。然后对每帧分别经过inception-v3的p re-train模型，得到特征，再对特征进行PCA，whitening以及压缩，得到Frame level data。将每个视频的所有Frame level data平均为1帧，得到Video level data。另外，audio的数据也是相同的流程进行处理。
![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/1.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/2.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/3.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/4.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/5.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/6.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/7.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/8.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/9.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/10.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/11.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/12.PNG)

![](https://github.com/NiyunZhou/The21-dayExpendables/blob/master/Slices/13.PNG)
