import os
import cv2
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import skimage
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, input_size, filepath, augment=True, training=True, isVal=False, facesDataSet=False,
                 isSpecific=False, mask_flist=None):
        super(Dataset, self).__init__()
        self.mask_flist = mask_flist
        self.augment = augment
        self.training = training
        self.isVal = isVal

        if facesDataSet == False and isSpecific == False:
            if isVal == False:
                with open('/kaggle/input/file-list/places365_train_standard.txt', 'r') as f:
                    self.imageNames = [line.split(None, 1)[0] for line in f]
                    self.imageNames = self.imageNames[:5000]
                    print(len(self.imageNames))
            else:
                with open('/kaggle/input/file-list/places365_val.txt', 'r') as f:
                    self.imageNames = [line.split(None, 1)[0] for line in f]
        self.isSpecific = isSpecific
        self.facesDataSet = facesDataSet
        self.data = self.load_flist(filepath)
        self.input_size = input_size
        #             self.edge_data = self.load_flist(edge_flist)
        if mask_flist is not None:
            self.mask_data = self.load_flist(mask_flist)
        self.nms = config.NMS
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_flist(self, flist):
        imagesPath = []
        if self.isSpecific == False:
            if self.facesDataSet:
                if os.path.isdir(flist):
                    return list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png')) + list(
                        glob.glob(flist + '/*.jpeg'))
            for filename in self.imageNames:
                if self.isVal == False:
                    imagesPath.append(os.path.join(flist, filename[1:]))
                else:
                    imagesPath.append(os.path.join(flist, filename))
        else:
            imagesPath.append(flist)
        #         To run on subset of images:
        #         if self.isVal == False:
        #             imagesPath = imagesPath[304896:]
        return imagesPath

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        img = cv2.imread(self.data[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) < 3:
            img = gray2rgb(img)
        if size != 0:
            img = self.resize(img, size, size)
        img_gray = rgb2gray(img)
        mask = self.load_mask(img, index)
        edge = self.load_edge(img_gray, index, mask)
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        # no edge
        if sigma == -1:
            return np.zeros(img.shape).astype(np.float)

        # random sigma
        if sigma == 0:
            sigma = random.randint(1, 4)

        return canny(img, sigma=sigma, mask=mask).astype(np.float)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return self.create_mask(imgw, imgh, imgw // 8, imgh // 8)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return self.create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = cv2.imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255  # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = cv2.imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]
        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, (height, width))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        #         plt.imshow(img)
        #         plt.show()
        img_t = F.to_tensor(img).float()
        return img_t

    def create_mask(self, width, height, mask_width, mask_height, x=None, y=None):
        mask = np.zeros((height, width))
        mask_x = x if x is not None else random.randint(0, width - mask_width)
        mask_y = y if y is not None else random.randint(0, height - mask_height)
        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item