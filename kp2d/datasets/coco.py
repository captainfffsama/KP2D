# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
import glob

from PIL import Image
from torch.utils.data import Dataset

def get_all_file_path(file_dir:str,filter_:tuple=('.jpg',)) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
            for filename in file_name_list \
            if os.path.splitext(filename)[1] in filter_ ]

class COCOLoader(Dataset):
    """
    Coco dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    data_transform : Function
        Transformations applied to the sample
    """
    def __init__(self, root_dir, data_transform=None):

        super().__init__()
        self.root_dir = root_dir

        self.files=[]

        if os.path.isdir(root_dir):
            self.files=get_all_file_path(root_dir)
        elif os.path.isfile(root_dir):
            with open(root_dir, 'r') as fr:
                self.files=[x.strip() for x in fr.readlines()]
        else:
            raise ValueError('训练数据集路径错误')
        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _read_rgb_file(self, filename):
        return Image.open(filename)

    def __getitem__(self, idx):

        filename = self.files[idx]
        image = self._read_rgb_file(filename)

        if image.mode == 'L':
            image_new = Image.new("RGB", image.size)
            image_new.paste(image)
            sample = {'image': image_new, 'idx': idx,'path': filename}
        else:
            sample = {'image': image, 'idx': idx,'path':filename}

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample
