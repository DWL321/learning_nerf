import torch.utils.data as data
import torch
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
import imageio
import json
import cv2

import numpy as np
import os

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()
#定义了一个匿名函数（lambda表达式），它接受一个参数t，然后返回一个4x4的张量（torch.Tensor），
#表示沿着z轴方向平移t单位的变换矩阵。.float()表示将张量转换为浮点数类型。

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()
#定义了一个匿名函数（lambda表达式），它接受一个参数phi，然后返回一个4x4的张量（torch.Tensor），
#表示绕着y轴方向旋转phi弧度的变换矩阵。.float()表示将张量转换为浮点数类型。


rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()
#定义了一个匿名函数（lambda表达式），它接受一个参数th，然后返回一个4x4的张量（torch.Tensor），
#表示绕着z轴方向旋转th弧度的变换矩阵。.float()表示将张量转换为浮点数类型。


def pose_spherical(theta, phi, radius):
    #这段代码的作用是定义一个函数pose_spherical，它接受三个参数theta，phi和radius，然后返回一个4x4的张量（torch.Tensor）
    #表示从球面坐标系（spherical coordinate system）到笛卡尔坐标系（Cartesian coordinate system）的变换矩阵。

    c2w = trans_t(radius)
    #c2w = trans_t(radius) 调用之前定义的函数trans_t，传入参数radius，得到一个沿着z轴方向平移radius单位的变换矩阵，
    #赋值给变量c2w。这一步相当于将球面坐标系的原点（0,0,0）移动到笛卡尔坐标系的（0,0,radius）处。

    c2w = rot_phi(phi/180.*np.pi) @ c2w
    #调用之前定义的函数rot_phi，传入参数phi/180.*np.pi，得到一个绕着y轴方向旋转phi度的变换矩阵
    #然后与变量c2w进行矩阵乘法（@表示矩阵乘法），得到新的变换矩阵，赋值给变量c2w。
    #这一步相当于将球面坐标系的z轴旋转到与笛卡尔坐标系的x轴平行。

    c2w = rot_theta(theta/180.*np.pi) @ c2w
    #调用之前定义的函数rot_theta，传入参数theta/180.*np.pi，得到一个绕着z轴方向旋转theta度的变换矩阵，
    #然后与变量c2w进行矩阵乘法（@表示矩阵乘法），得到新的变换矩阵，赋值给变量c2w。
    #这一步相当于将球面坐标系的x轴旋转到与笛卡尔坐标系的y轴平行。

    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    #创建一个4x4的张量（torch.Tensor），表示一个对角线为（-1,1,1,1）的对角矩阵（diagonal matrix），
    #然后与变量c2w进行矩阵乘法（@表示矩阵乘法），得到新的变换矩阵，赋值给变量c2w。
    #这一步相当于将球面坐标系的y轴和z轴交换，并且将x轴反向。

    #最后返回变量c2w作为函数的输出。
    return c2w


#minify函数用于缩小图像的尺寸和分辨率。
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    # images_4
    # images_8 目录
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            # 不存在对应的目录
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            # 不存在对应的目录
            needtoload = True

    # 如果存在那些目录, 那么这里就返回
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        # 执行一个shell命令
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
#定义一个名为 _load_data 的函数，它接受五个参数：basedir, factor, width, height, load_imgs。
# basedir 是一个字符串，表示图像文件所在的基本目录；factor 是一个整数，表示要缩小图像的倍数；
# width 和 height 是两个整数，表示要缩小图像的宽度和高度；load_imgs 是一个布尔值，表示是否要加载图像数据。
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
#函数首先从 basedir 目录下的 poses_bounds.npy 文件中加载姿态和边界数据，
# 将其转换为合适的形状和维度，并分别赋值给 poses 和 bds 两个变量。
    img0 = [os.path.join(basedir, 'images_8', f) for f in sorted(os.listdir(os.path.join(basedir, 'images_8'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    #函数获取 basedir 目录下的 images 子目录中的第一个图像文件，并读取其形状信息，存储在 sh 变量中。
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    #函数根据 factor, width, height 三个参数中的任意一个确定要缩小图像的比例和尺寸，
# 并调用 _minify 函数进行缩小操作。函数根据缩小后的图像目录的名称赋值给 sfx 变量。
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    #函数检查缩小后的图像目录是否存在，如果不存在，就打印出错误信息并返回。
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    #函数获取缩小后的图像目录中的所有图像文件，并按照文件名排序，存储在 imgfiles 列表中。
# 如果姿态数据的数量和图像文件的数量不匹配，就打印出错误信息并返回。
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    #函数更新 poses 变量中的最后一行数据，将其设置为缩小后的图像形状和比例信息。
    if not load_imgs:
        return poses, bds
    #如果 load_imgs 参数为 False，函数就只返回 poses 和 bds 两个变量。
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    #如果 load_imgs 参数为 True，函数就定义一个名为 imread 的内部函数，用于根据文件后缀名读取图像数据，
# 并将其归一化到 [0, 1] 区间。函数使用 imread 函数读取 imgfiles 列表中的所有图像文件，并将其堆叠成一个四维数组，
# 存储在 imgs 变量中。
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs
    #函数打印出加载图像数据的信息，并返回 poses, bds, imgs 三个变量。
    
            
            
def normalize(x):
    return x / np.linalg.norm(x)
#这个函数的作用是将一个向量或矩阵进行归一化，也就是使其长度（范数）为1。
# 具体来说，它会先用np.linalg.norm(x)计算出x的范数，然后用x除以这个范数，得到一个新的向量或矩阵，它的范数为1，但方向不变。
# 这样做的好处是可以消除不同向量或矩阵之间的量纲差异，便于进行比较或计算。

#np.linalg.norm(x)是一个NumPy内置的函数，它可以根据不同的参数返回不同类型的范数。
# 范数是一种衡量向量或矩阵大小的方法，有多种定义方式。你可以参考这里了解更多关于np.linalg.norm(x)的用法和范数的定义


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def spherify_poses(poses, bds):
#这段代码的作用是将一组相机姿态（poses）进行球面化（spherify），
# 使得它们能够围绕一个物体的中心点以一定的半径和高度分布在一个球面上，从而方便模型进行360度的渲染和重建。
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
#首先，定义一个函数p34_to_44，用于将3x4的相机姿态矩阵扩展为4x4的齐次坐标矩阵，即在最后一行添加[0,0,0,1]。
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]
#然后，从poses中提取出每个相机的方向向量（rays_d）和原点位置（rays_o），它们分别对应于poses的第三列和第四列。
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist
#接着，定义一个函数min_line_dist，用于计算一组射线与其平均位置之间的最小距离。
# 这个函数的输入是射线的原点（rays_o）和方向（rays_d），输出是最小距离处的空间点（pt_mindist）。
# 这个函数的实现是基于最小二乘法，求解每条射线与平均位置之间的垂直距离的平方和最小化问题。
    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)
#然后，调用min_line_dist函数，得到最小距离处的空间点pt_mindist，将其作为物体的中心点（center）。
# 同时，计算poses中每个相机原点与中心点之间的向量的平均值，将其作为球面上方向（up）。
    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
#接着，根据up向量构造一个新的相机姿态矩阵c2w，表示从球面上某一点到世界坐标系的变换。
# 具体地，先将up向量归一化为vec0，然后用vec0与一个随机向量叉乘得到vec1，并归一化。
# 再用vec0与vec1叉乘得到vec2，并归一化。最后将vec1, vec2, vec0, center按列拼接起来得到c2w矩阵。
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])
#然后，将c2w矩阵的逆矩阵左乘poses中每个相机姿态矩阵，并用p34_to_44函数扩展为4x4矩阵，得到poses_reset。
# 这个操作相当于将所有相机姿态都变换到以c2w为参考系的坐标系下。
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
#然后，计算poses_reset中每个相机原点到原点[0,0,0]的欧氏距离，并求平均值，得到半径rad。
# 然后将poses_reset中每个相机原点除以rad，得到归一化后的相机原点。同时，将输入参数bds也除以rad，并更新rad的值。
# 这个操作相当于将所有相机姿态都缩放到一个单位球面上。
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
#然后，计算poses_reset中每个相机原点在z轴上的分量，并求平均值，得到高度zh。然后根据zh和rad计算出球面上圆周的半径radcircle。

    new_poses = []

    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)
#然后，定义一个空列表new_poses，用于存储新生成的相机姿态。
# 然后在[0, 2*pi]区间内均匀地取120个角度值th，并根据th, radcircle和zh计算出对应的相机原点camorigin。
# 然后定义一个向量up为[0,0,-1]。然后根据camorigin和up构造一个新的相机姿态矩阵p，并添加到new_poses列表中。
# 这个操作相当于在球面上生成了120个等间距分布的相机姿态。


    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
#最后，将new_poses列表转换为numpy数组，并在最后一列添加poses中的第四列（表示相机的内参）。
#同时，将poses_reset也在最后一列添加poses中的第四列。然后返回poses_reset, new_poses, bds作为结果。

#这段代码是神经辐射场（NeRF）模型中处理LLFF数据集时使用的函数之一。
# 它可以使得不同场景下的相机姿态具有一致性和可比性，从而方便模型进行渲染和重建。





class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        #初始化参数
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split  # train or test
        self.precrop_frac = cfg.task_arg.precrop_frac
        self.precrop_iters = cfg.task_arg.precrop_iters
        self.use_single_view = cfg.train.single_view
        self.num_iter_train = 0
        self.white_bkgd = cfg.task_arg.white_bkgd
        self.use_batching = not cfg.task_arg.no_batching
        self.batch_size = cfg.task_arg.N_rays
        self.render_only = True
        self.factor = 8
        self.recenter = True
        self.bd_factor = .75
        self.spherify = False
        self.path_zflat = False

        poses, bds, imgs = _load_data(self.data_root, factor=self.factor)
        print('Loaded', self.data_root, bds.min(), bds.max())
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        self.images = imgs
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        sc = 1. if self.bd_factor is None else 1./(bds.min() * self.bd_factor)
        poses[:,:3,3] *= sc
        bds *= sc
        if self.recenter:
            poses = recenter_poses(poses)
        if self.spherify:
            poses, self.render_poses, bds = spherify_poses(poses, bds)
        else:
            c2w = poses_avg(poses)
            print('recentered',c2w.shape)
            print(c2w[:3,:4])

            up = normalize(poses[:, :3, 1].sum(0))
            close_depth, inf_depth = bds.min()*.9, bds.max()*5.
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focal = mean_dz

            shrink_factor = .8
            zdelta = close_depth * .2
            tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
            rads = np.percentile(np.abs(tt), 90, 0)
            c2w_path = c2w
            N_views = 120
            N_rots = 2
            if self.path_zflat:
    #             zloc = np.percentile(tt, 10, 0)[2]
                zloc = -close_depth * .1
                c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
                rads[2] = 0.
                N_rots = 1
                N_views/=2
            render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        render_poses = np.array(render_poses).astype(np.float32)
        c2w = poses_avg(poses)
        print('Data:')
        print(poses.shape, self.images.shape, bds.shape)

        dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
        i_test = np.argmin(dists)
        print('HOLDOUT view is', i_test)
        
        imgs = self.images.astype(np.float32)
        self.num_imgs = imgs.shape[0]
        self.imgs = torch.from_numpy(imgs)
        poses = poses.astype(np.float32)
        self.poses = torch.from_numpy(poses)
        
        self.bds = bds
        self.near = np.ndarray.min(self.bds) * .9
        self.far = np.ndarray.max(self.bds) * 1.
        
        self.render_poses = render_poses
        self.i_test = i_test
        self.focal = focal
        hwf= self.poses[0, :3, -1]
        self.H, self.W, self.focal = hwf
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.focal]
        self.poses = self.poses[:,:3,:4]
        print("H:",self.H, "W:",self.W, "focal:",self.focal)
        self.K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ])

        dH = int(self.H // 2 * self.precrop_frac)
        dW = int(self.W // 2 * self.precrop_frac)
        self.coords_center = torch.stack(
            torch.meshgrid(
                torch.linspace(self.H // 2 - dH, self.H // 2 + dH - 1, 2 * dH),
                torch.linspace(self.W // 2 - dW, self.W // 2 + dW - 1, 2 * dW), indexing='ij',
            ), -1
        )
        self.coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, self.H - 1, self.H),
                torch.linspace(0, self.W - 1, self.W), indexing='ij',
            ), -1
        )

        rays_o = []
        rays_d = []
        for i in range(self.num_imgs):
            ray_o, ray_d = self.get_rays(self.H, self.W, self.K, self.poses[i, :3, :4])
            rays_d.append(ray_d)# (H, W, 3)
            rays_o.append(ray_o)# (H, W, 3)
        self.rays_o = torch.stack(rays_o)#(num_imgs, H, W, 3)
        self.rays_d = torch.stack(rays_d)#(num_imgs, H, W, 3)
        self.render_rays_o, self.render_rays_d = self.get_render_rays()

    def __getitem__(self, index):
        index = 0 if self.use_single_view else index
        
        ray_o = self.rays_o[index].reshape(-1, 3)
        ray_d = self.rays_d[index].reshape(-1, 3)
        rgb = self.imgs[index].reshape(-1, 3) 
        
        ret = {'ray_o': ray_o, 'ray_d': ray_d, 'rgb': rgb, 'near':self.near, 'far':self.far }

        ret.update({'meta':
            {
                'H': self.H,
                'W': self.W,
                'ratio': self.input_ratio,
                'N_rays': self.batch_size,
                'id': index,
                'num_imgs': self.num_imgs
            }
        })
        return ret

    def __len__(self):
        
        #return self.num_imgs 
        return 108

    #获得光线的起点和方向数据。
    def get_rays(self, H, W, K, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return rays_o, rays_d

    #自己设定视角（新视角），来获取新视角的光线起点和方向，以便于进行渲染。
    def get_render_rays(self):
        '''
        #self.render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]],0)
        num_poses = 40
        angles = np.linspace(-180, 180, num_poses+1)[:-1]
        #`num_poses`表示姿势数量，然后使用`np.linspace`生成角度值的数组`angles`，并且通过切片操作`[:-1]`去掉了最后一个元素。
        render_poses= []#创建了一个空列表`render_poses`，
        for angle in angles:#通过for循环遍历`angles`中的每个角度值。
            pose = pose_spherical(angle, -30.0, 4.0)
            render_poses.append(pose)
            #在每次迭代中，我们调用`pose_spherical`函数生成对应角度的姿势，并将其添加到`render_poses`列表中。
        self.render_poses = np.stack(render_poses, axis=0)
        #使用`np.stack`函数将`render_poses`列表中的姿势数组沿着新的轴（`axis=0`）堆叠起来
        #并将结果赋值给`self.render_poses`属性。
        '''

        self.render_poses = torch.from_numpy(self.render_poses)
        render_rays_o, render_rays_d = [], []

        for i in range(self.render_poses.shape[0]):
            render_ray_o, render_ray_d = self.get_rays(self.H, self.W, self.K, self.render_poses[i, :3, :4])
            render_rays_o.append(render_ray_o)
            render_rays_d.append(render_ray_d)
        render_rays_o = torch.stack(render_rays_o)
        render_rays_d = torch.stack(render_rays_d)
        return render_rays_o, render_rays_d