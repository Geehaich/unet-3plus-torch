import torch
import torch.nn as tnn
import os

import torchvision.transforms
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import random

import json
import cv2
import numpy as np

class SingleClassDataset(torch.utils.data.Dataset) :
    """loads input images and binary masks from a root directory, returns a tuple containing
      the input image, the mask, and a 1-hot encoded vector indicating if the image is a negative or positive.

      prerequesites :
        -the structure of the root directory is  root
                                                   '-->Images
                                                   '-->Masks
        -every mask file and image file has the same extension
        """
    def __init__(self,root_directory,
                 image_shape = [256,256],
                 image_ext = ".jpg",
                 mask_ext= ".png",
                 _device = 0):

        super().__init__()
        im_dir = root_directory+'Images/'
        mask_dir = root_directory+'Masks/'
        self.image_list = [im_dir+f for f in sorted(os.listdir(im_dir)) if image_ext in f]
        self.mask_list = [f.replace(im_dir,mask_dir).replace(image_ext,mask_ext) for f in self.image_list]

        self.resizer = torchvision.transforms.Resize(image_shape)
        self.device = torch.device(_device) if _device >= 0  else torch.device("cpu")

    def __getitem__(self, item):

        #load normalized image
        im = read_image(self.image_list[item]).float()
        im /= im.max()
        mask = read_image(self.mask_list[item]).float()

        one_hot_mask = torch.zeros([2,*mask.shape[1:]],dtype=torch.float)

        

        one_hot_mask[1,:] = (mask>0.5).float()
        one_hot_mask[0,:] = (mask<0.5).float()
        mask = one_hot_mask



        im = self.resizer(im)
        mask = self.resizer(mask)

        #random augmentations

        if torch.randint(1,100,[1]).item()<50 :
            im = TF.hflip(im)
            mask = TF.hflip(mask)
        if torch.randint(1,100,[1]).item()<50 :
            im = TF.vflip(im)
            mask = TF.vflip(mask)



        pres = (torch.count_nonzero(mask)!=0).float().item()
        one_hot_presence = torch.Tensor([1-pres,pres])

        im = im.to(self.device)
        mask = mask.to(self.device)
        one_hot_presence = one_hot_presence.to(self.device)


        return im,mask,one_hot_presence

    def __len__(self):
        return len(self.image_list)

class COCODataset(torch.utils.data.Dataset) :

    def __init__(self,rootdir,
                 image_shape = [256,256],
                 _device=0):

        super().__init__()
        with open(rootdir+'_annotations.coco.json') as fopen :
            Parsed = json.loads(fopen.read())

        self.root = rootdir
        self.image_list = [fpath for fpath in Parsed["images"]]
        self.categories =  Parsed["categories"]
        self.annotations = Parsed["annotations"]

        self.resizer = torchvision.transforms.Resize(image_shape)

        self.device = torch.device(_device) if _device >= 0  else torch.device("cpu")



    def __getitem__(self, item):

        #normalize image
        im = read_image(self.root+self.image_list[item]["file_name"]).float()
        im /= im.max()

        #make one hot mask. start background at 1s and draw black on it later
        one_hot_mask = np.zeros([len(self.categories),*im.shape[-2:]],dtype=np.uint8)
        one_hot_mask[0,...] = 1
        presence_vector = torch.zeros([len(self.categories)])
        presence_vector[0] = 1

        #get all annotations corresponding to this image
        image_annotations = [annot_dict for annot_dict in self.annotations if annot_dict['image_id']==self.image_list[item]['id']]

        #draw mask and populate presence vector
        for annotation in image_annotations :
            cls_index = annotation["category_id"]
            polypoints = np.reshape(annotation["segmentation"][0],(len(annotation["segmentation"][0])//2, 2)).astype(np.int32)
            presence_vector[cls_index] = 1
            one_hot_mask[cls_index,...] = cv2.fillPoly(one_hot_mask[cls_index,...],[polypoints],1)
            one_hot_mask[0,...] = cv2.fillPoly(one_hot_mask[0,...],[polypoints],0)

        one_hot_mask = torch.Tensor(one_hot_mask).float()

        im = self.resizer(im)
        one_hot_mask=self.resizer(one_hot_mask)

        im = im.to(self.device)
        one_hot_mask = one_hot_mask.to(self.device)
        presence_vector = presence_vector.to(self.device)

        return im,one_hot_mask,presence_vector

    def __len__(self):
        return len(self.image_list)
