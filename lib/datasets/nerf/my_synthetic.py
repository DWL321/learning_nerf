import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2

def trans_t(t):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,t],
        [0,0,0,1],
    ], dtype=np.float32)

def rot_phi(phi):
    return np.array([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1],
    ], dtype=np.float32)

def rot_theta(th):
    return np.array([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1],
    ], dtype=np.float32)

def pose_spherical(theta, phi, radius):
    """
    Input:
        @theta: [-180, +180]，间隔为 9
        @phi: 固定值 -30
        @radius: 固定值 4
    Output:
        @c2w: 从相机坐标系到世界坐标系的变换矩阵
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

class Dataset(data.Dataset):
    # init函数负责从磁盘中load指定格式的文件，计算并存储为特定形式
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        cams = kwargs['cams']
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays

        # read image
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames']:
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))

        img = imageio.imread(image_paths[view])/255.
        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        if self.input_ratio != 1.:
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
        # set image
        self.img = np.array(img).astype(np.float32)
        # set uv
        H, W = img.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1)
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)


    # getitem函数负责在运行时提供给网络一次训练需要的输入，以及groundtruth的输出。 例如对NeRF，分别是1024条rays以及1024个RGB值。
    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(len(self.uv), self.batch_size, replace=False)
            uv = self.uv[ids]
            rgb = self.img.reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = self.img.reshape(-1, 3)
        ret = {'uv': uv, 'rgb': rgb} # input and output. they will be sent to cuda
        ret.update({'meta': {'H': self.img.shape[0], 'W': self.img.shape[1]}}) # meta means no need to send to cuda
        return ret

    # len函数是训练或者测试的数量。getitem函数获得的index值通常是[0, len-1]
    def __len__(self):
        # we only fit 1 images, so we return 1
        return 1
