# -*- coding: utf-8 -*-

# @Description: 
# @Author: CaptainHu
# @Date: 2020-12-24 16:41:29
# @LastEditors: CaptainHu

from pathlib import Path

import ipdb

import context

import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class EvalDataset(Dataset):
    """
    Eval dataset class.
    Note: output_shape = (output_width, output_height)
    Note: this returns Pytorch tensors, resized to output_shape (if specified)
    Note: the homography will be adjusted according to output_shape.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    use_color : bool
        Return color images or convert to grayscale.
    data_transform : Function
        Transformations applied to the sample
    output_shape: tuple
        If specified, the images and homographies will be resized to the desired shape.
    type: str
        Dataset subset to return from ['i', 'v', 'all']: 
        i - illumination sequences
        v - viewpoint sequences
        all - all sequences
    """
    def __init__(self, root_dir, use_color=True, data_transform=None, output_shape=None, type='all',ext='.ppm'):

        super().__init__()
        self.type = type
        self.root_dir = root_dir
        self.data_transform = data_transform
        self.output_shape = output_shape
        self.use_color = use_color
        base_path = Path(root_dir)
        folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
        image_paths = []
        warped_image_paths = []
        homographies = []
        for path in folder_paths:
            if self.type == 'i' and path.stem[0] != 'i':
                continue
            if self.type == 'v' and path.stem[0] != 'v':
                continue
            num_images = 5
            file_ext = ext
            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(path, "1" + file_ext)))
                warped_image_paths.append(str(Path(path, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
        self.files = {'image_paths': image_paths, 'warped_image_paths': warped_image_paths, 'homography': homographies}

    @staticmethod
    def scale_homography(homography, original_scale, new_scale, pre):
        scales = np.divide(new_scale, original_scale)
        if pre:
            s = np.diag(np.append(scales, 1.))
            homography = np.matmul(s, homography)
        else:
            sinv = np.diag(np.append(1. / scales, 1.))
            homography = np.matmul(homography, sinv)
        return homography

    def __len__(self):
        return len(self.files['image_paths'])

    def __getitem__(self, idx):

        def _read_image(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.use_color:
                return img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return gray

        image = _read_image(self.files['image_paths'][idx])

        warped_image = _read_image(self.files['warped_image_paths'][idx])
        homography = np.array(self.files['homography'][idx])
        sample = {'image': image, 'warped_image': warped_image, 'homography': homography, 'index' : idx,'w_img_path':self.files['warped_image_paths'][idx]}

        # Apply transformations
        '''
        H_{new}=S_{b_scale} H S^{-1}_{a_scale}
        '''
        if self.output_shape is not None:
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         sample['image'].shape[:2][::-1],
                                                         self.output_shape,
                                                         pre=False)
            sample['homography'] = self.scale_homography(sample['homography'],
                                                         sample['warped_image'].shape[:2][::-1],
                                                         self.output_shape,
                                                         pre=True)
            for key in ['image', 'warped_image']:
                sample[key+'_scale'] =self.get_scale(sample[key].shape[:2][::-1],return_inverse=True)
                sample[key] = cv2.resize(sample[key], self.output_shape)
                if self.use_color is False:
                    sample[key] = np.expand_dims(sample[key], axis=2)

        transform = transforms.ToTensor()
        for key in ['image', 'warped_image']:
            sample[key+'_t'] = transform(sample[key]).type('torch.FloatTensor')
        return sample
    
    def get_scale(self,ori_size,return_inverse=False):
        img_scale_H=np.eye(3)
        if ori_size !=self.output_shape:
            img_scale_H=np.divide(np.array(self.output_shape),ori_size)
            if return_inverse:
                img_scale_H=np.diag(np.append(1./img_scale_H,1.))
            else:
                img_scale_H=np.diag(np.append(img_scale_H,1.))
        return img_scale_H

if __name__=="__main__":
    dataset_path=r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/kp2d/HPatches'
    eval_dataset=EvalDataset(dataset_path,True,output_shape=(640,480))
    from torch.utils.data import DataLoader
    data_loader = DataLoader(eval_dataset,
                                batch_size=0,
                                pin_memory=False,
                                shuffle=False,
                                num_workers=1,
                                worker_init_fn=None,
                                sampler=None)
    for sample in data_loader:
        import ipdb; ipdb.set_trace()
