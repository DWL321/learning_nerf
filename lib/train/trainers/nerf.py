import torch
import torch.nn as nn
from lib.networks.nerf.renderer import volume_renderer
#首先，代码导入了需要使用的库和模块，包括`torch`和`torch.nn`，以及自定义模块`volume_renderer`。

class NetworkWrapper(nn.Module):
#接下来定义了`NetworkWrapper`类，它继承自`nn.Module`类，表示它是一个PyTorch模型。
    def __init__(self, net, train_loader):
        #在`__init__`方法中，进行了一些初始化操作：
        super(NetworkWrapper, self).__init__()
        self.net = net #self.net，这里的net来自于main()函数中的network = make_network(cfg)
        #network = make_network(cfg)最终是network作为Network类的一个实例。
        #`self.net = net`：将传入的`net`参数赋值给`self.net`，表示该模型包装器将使用这个神经网络模型。
        #这里的net暂时是与render渲染无关的哦
        self.renderer = volume_renderer.Renderer(self.net)
        #self.renderer = volume_renderer.Renderer(self.net)`：
        #创建了一个名为`renderer`的`volume_renderer.Renderer`对象，
        #传入的参数为`self.net`，表示使用该网络模型进行渲染操作。
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        #定义了一个名为`img2mse`的匿名函数，用于计算图像均方误差（Mean Squared Error，MSE）。
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        #定义了一个名为`mse2psnr`的匿名函数，
        #用于将均方误差（MSE）转换为峰值信噪比（Peak Signal-to-Noise Ratio，PSNR）。
        self.acc_crit = torch.nn.functional.smooth_l1_loss


    def forward(self, batch):
        ret = self.renderer.render(batch)
        #这里返回的ret非常重要，它包含了Renderer类的主要函数，包括了分层采样等trick，可以用于获取rgb预测值
        scalar_stats = {}
        loss = 0

        img_loss = self.img2mse(ret['rgb_map'], batch['rgb'])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
