# Learning NeRF
## 环境配置
```
conda create -n learning_nerf python=3.9
conda activate learning_nerf
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
## 从Image fitting demo来学习这个框架
### MLP
[多层感知机](https://blog.csdn.net/JasonH2021/article/details/131021534):层与层之间全连接,使用激活函数给神经元引入非线性因素，可以完成分类、回归和聚类等任务。
### 任务定义
训练一个MLP，将某一张图像的像素坐标作为输入, 输出这一张图像在该像素坐标的RGB value。
 
**Training**

```
python train_net.py --cfg_file configs/img_fit/lego_view0.yaml
```

**Evaluation**

```
python run.py --type evaluate --cfg_file configs/img_fit/lego_view0.yaml
```

**查看loss曲线**

```
tensorboard --logdir=data/record --bind_all
//在浏览器打开localhost:6006
```
### 遇到的问题
**训练到epoch 9评估器evaluate时报错**
```
Traceback (most recent call last):
  File "/home/dwl/anaconda3/envs/learning_nerf/lib/python3.9/site-packages/PIL/Image.py", line 3070, in fromarray
    mode, rawmode = _fromarray_typemap[typekey]
KeyError: ((1, 1, 3), '<f4')

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/dwl/Desktop/reasearch/nerf_recurrent/learning_nerf/train_net.py", line 116, in <module>
    main()
  File "/home/dwl/Desktop/reasearch/nerf_recurrent/learning_nerf/train_net.py", line 108, in main
    train(cfg, network)
  File "/home/dwl/Desktop/reasearch/nerf_recurrent/learning_nerf/train_net.py", line 67, in train
    trainer.val(epoch, val_loader, evaluator, recorder)
  File "/home/dwl/Desktop/reasearch/nerf_recurrent/learning_nerf/lib/train/trainers/trainer.py", line 105, in val
    image_stats_ = evaluator.evaluate(output, batch)
  File "lib/evaluators/img_fit.py", line 31, in evaluate
    imageio.imwrite(save_path, img_utils.horizon_concate(gt_rgb, pred_rgb))
  File "/home/dwl/anaconda3/envs/learning_nerf/lib/python3.9/site-packages/imageio/v2.py", line 397, in imwrite
    return file.write(im, **kwargs)
  File "/home/dwl/anaconda3/envs/learning_nerf/lib/python3.9/site-packages/imageio/plugins/pillow.py", line 444, in write
    pil_frame = Image.fromarray(frame, mode=mode)
  File "/home/dwl/anaconda3/envs/learning_nerf/lib/python3.9/site-packages/PIL/Image.py", line 3073, in fromarray
    raise TypeError(msg) from e
TypeError: Cannot handle this data type: (1, 1, 3), <f4
```
需要将浮点数数组gt_rgb 和 pred_rgb转化为整数数组：
```
        pred_rgb = output['rgb'][0].reshape(H, W, 3).detach().cpu().numpy()
        pred_rgb = pred_rgb.astype(np.uint8)
        gt_rgb = batch['rgb'][0].reshape(H, W, 3).detach().cpu().numpy()
        gt_rgb = gt_rgb.astype(np.uint8)
```
**evaluate结果为纯黑图片，是因为图像像素值归一化过但是保存结果时没有恢复**
```
//修改learning_nerf/lib/evaluators/img_fit.py为
        pred_rgb = output['rgb'][0].reshape(H, W, 3).detach().cpu().numpy()*255
        pred_rgb = pred_rgb.astype(np.uint8)
        gt_rgb = batch['rgb'][0].reshape(H, W, 3).detach().cpu().numpy()*255
        gt_rgb = gt_rgb.astype(np.uint8)
```
## 复现NeRF
### 配置文件
修改configs/nerf/nerf.yaml
### 创建dataset： lib.datasets.nerf.synthetic.py和lib.datasets.nerf.synthetic_path.py

核心函数包括：init, getitem, len.

init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式。

getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。
例如对NeRF，分别是1024条rays以及1024个RGB值。

len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]。


#### debug：

```
python run.py --type dataset --cfg_file configs/nerf/nerf.yaml
```

### 创建network:

核心函数包括：init, forward.

init函数负责定义网络所必需的模块，forward函数负责接收dataset的输出，利用定义好的模块，计算输出。例如，对于NeRF来说，我们需要在init中定义两个mlp以及encoding方式，在forward函数中，使用rays完成计算。

创建lib/networks/nerf/network.py、lib/networks/nerf/renderer/make_renderer.py、lib/networks/nerf/renderer/nerf_net_utils.py、100644 lib/networks/nerf/renderer/volume_renderer.py

#### debug：

```
python run.py --type network --cfg_file configs/nerf/nerf.yaml
```
**出现RuntimeError：No CUDA GPUs are available报错**
```
//在.cuda()前面
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    torch.cuda.current_device()
    torch.cuda._initialized = True
```
### loss模块和evaluator模块

创建lib/evaluators/nerf.py和lib/train/losses/nerf.py、lib/train/trainers/nerf.py

debug方式分别为：

```
python train_net.py --cfg_file configs/nerf/nerf.yaml
```

```
python run.py --type evaluate --cfg_file configs/nerf/nerf.yaml
```
### 参考资料
+ [NeRF源码解析](https://www.bilibili.com/video/BV1d841187tn/?share_source=copy_web&vd_source=82f2d2d3d2d3b3112e473c0a443cc278)