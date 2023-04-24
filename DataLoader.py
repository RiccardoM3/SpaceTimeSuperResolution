import os
import random
import pickle
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def create_dataloader(dataset):
    num_workers = 4 # per GPU
    batch_size = 7
    shuffle = True
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, drop_last=True,
                                           pin_memory=False)

def create_dataset(train_path):
    return VimeoDataset(train_path)


'''
Vimeo7 dataset
'''
class VimeoDataset(Dataset):
    '''
    Reading the training Vimeo dataset
    key example: train/00001/0001/im1.png
    '''

    def __init__(self, train_path):
        super(VimeoDataset, self).__init__()
        self.scale = 4
        self.num_frames = 7
        self.HR_crop_size = 256
        self.LR_crop_size = self.HR_crop_size // self.scale
        self.HR_image_shape = (3, 256, 448)
        self.LR_image_shape = (3, self.HR_image_shape[1] // self.scale, self.HR_image_shape[2] // self.scale)

        self.LR_num_frames = 1 + self.num_frames // 2
        assert self.LR_num_frames > 1, 'Error: Not enough LR frames to interpolate'

        self.LR_index_list = [i * 2 for i in range(self.LR_num_frames)]

        self.HR_root = os.path.join(train_path, "sequences", "train")
        self.LR_root = os.path.join(train_path, "sequences_LR", "train")

        # Load image keys
        cache_keys = 'Vimeo_keys.pkl'
        self.HR_paths = list(pickle.load(open('{}'.format(cache_keys), 'rb'))['keys'])

        assert self.HR_paths, 'Error: HR path is empty.'

    def read_img(self, path):
        """Read image using cv2.

        Args:
            path (str): path to the image.

        Returns:
            array: (H, W, C) BGR image. 
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    # TODO: use GPU to flip horizontally, flip vertically, flip temorally, and rotate each 50% of the time.
    def augment(self, img_list, hflip=True, rot=True):
        return img_list

    def __getitem__(self, index):
        key = self.HR_paths[index]
        name_a, name_b = key.split('_')

        # Get frame list
        HR_frames_list = list(range(1, self.num_frames + 1))
        if random.random() < 0.5:
            HR_frames_list.reverse()
        LR_frames_list = [HR_frames_list[i] for i in self.LR_index_list]

        # Get HR images
        img_HR_list = []
        for v in HR_frames_list:           
            img_HR = self.read_img(os.path.join(self.HR_root, name_a, name_b, 'im{}.png'.format(v)))
            img_HR_list.append(img_HR)
                
        # Get LR images
        img_LR_list = []
        for v in LR_frames_list:
            img_LR = self.read_img(os.path.join(self.LR_root, name_a, name_b, 'im{}.png'.format(v)))
            img_LR_list.append(img_LR)

        # TODO: perform below operations in GPU
        _, H, W = self.LR_image_shape
        # Randomly crop
        rnd_h = random.randint(0, max(0, H - self.LR_crop_size))
        rnd_w = random.randint(0, max(0, W - self.LR_crop_size))
        img_LR_list = [v[rnd_h:rnd_h + self.LR_crop_size, rnd_w:rnd_w + self.LR_crop_size, :] for v in img_LR_list]
        rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
        img_HR_list = [v[rnd_h_HR:rnd_h_HR + self.HR_crop_size, rnd_w_HR:rnd_w_HR + self.HR_crop_size, :] for v in img_HR_list]

        # Augmentation - flip, rotate
        img_list = img_LR_list + img_HR_list
        img_list = self.augment(img_list, True, True)
        img_LR_list = img_list[0:-self.num_frames]
        img_HR_list = img_list[-self.num_frames:]

        # Stack LR images to NHWC, N is the frame number
        img_LRs = np.stack(img_LR_list, axis=0)
        img_HRs = np.stack(img_HR_list, axis=0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_HRs = img_HRs[:, :, :, [2, 1, 0]]
        img_LRs = img_LRs[:, :, :, [2, 1, 0]]
        img_HRs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HRs, (0, 3, 1, 2)))).float()
        img_LRs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LRs, (0, 3, 1, 2)))).float()

        return {'LRs': img_LRs, 'HRs': img_HRs, 'key': key}

    def __len__(self):
        return len(self.HR_paths)
