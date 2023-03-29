import torch
import torch.nn as tnn
import os
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import random


class SingleClassDataset(torch.utils.data.Dataset) :
    """loads input images and masks from a root directory, returns a tuple containing
      the input image, the mask, and a 1-hot encoded vector indicating if the image is a negative or positive.

      prerequesites :
        -the structure of the root directory is  root
                                                   '-->Images
                                                   '-->Masks
        -every mask file and image file has the same extension
        """
    def __init__(self,root_directory,image_ext = ".jpg",mask_ext= ".png"):

        super().__init__()
        im_dir = root_directory+'Images'
        mask_dir = root_directory+'Images'
        self.image_list = [im_dir+f for f in sorted(os.listdir(im_dir)) if image_ext in f]
        self.mask_list = [f.replace(im_dir,mask_dir).replace(image_ext,mask_ext) for f in self.image_list]

    def __getitem__(self, item):

        im = read_image(self.image_list[item]).unsqueeze(0).float()
        mask = read_image(self.mask_list[item]).unsqueeze(0).float()

        im = tnn.functional.normalize(im,1)
        mask = tnn.functional.normalize(mask,1)

        #random augmentations

        if torch.randint(1,100,[1]).item()<50 :
            im = TF.hflip(im)
            mask = TF.hflip(mask)
        if torch.randint(1,100,[1]).item()<50 :
            im = TF.vflip(im)
            mask = TF.vflip(mask)

        if torch.randint(1,100,[1]).item()<15 :
            im = TF.gaussian_blur(im,[3,3],0.03)

    def __len__(self):
        return len(self.image_list)

