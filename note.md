# Learning NeRF
## 环境配置
conda create -n learning_nerf python=3.9

source activate learning_nerf

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
## 从Image fitting demo来学习这个框架
### MLP
[多层感知机](https://blog.csdn.net/JasonH2021/article/details/131021534):层与层之间全连接,使用激活函数给神经元引入非线性因素，可以完成分类、回归和聚类等任务。
### 

## 复现NeRF
### 参考资料
+ [NeRF源码解析](https://www.bilibili.com/video/BV1d841187tn/?share_source=copy_web&vd_source=82f2d2d3d2d3b3112e473c0a443cc278)
###