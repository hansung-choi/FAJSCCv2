import torchvision
from glob import glob
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
import os, logging
from hydra import initialize, compose
import torch
from hydra.utils import instantiate
import albumentations as A 
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision import datasets, transforms
from PIL import Image
import random
import numpy as np
from os.path import join
from os import listdir
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

def DataMaker(cfg: DictConfig):
    data = None
    cfg.input_resolution = (128,128)
    if cfg.data_mode == "train":
        data = TrainDataMaker(cfg)
    elif cfg.data_mode == "test":
        data = TestDataMaker(cfg)
    else:
        raise ValueError(f'{cfg.data_mode} is not implemented yet')
    return data

def TrainDataMaker(cfg: DictConfig):
    data = None
    cfg.input_resolution = (128,128)
    if cfg.data_info.data_name == "DIV2K":
        data = DIV2Kdataset_train(cfg)
    elif cfg.data_info.data_name == "Flickr30k":
        data = Flickr30kdataset_train(cfg)
    else:
        raise ValueError(f'{cfg.data_info.data_name} is not implemented yet')
    return data
    
def TestDataMaker(cfg: DictConfig):
    data = None
    #print("cfg.test_data:",cfg.test_data)
    cfg.input_resolution = (128,128)
    
    if cfg.test_data == "DIV2K":
        data = DIV2Kdataset_test(cfg)
    elif cfg.test_data == "Kodak":
        data = Kodakdataset_test(cfg)
    else:
        raise ValueError(f'{cfg.test_data} is not implemented yet')
    return data


class DIV2Kdataset_train():
    def __init__(self,cfg: DictConfig):
        self.data_name = cfg.data_info.data_name
        self.batch_size = cfg.data_info.batch_size
        self.num_workers = cfg.data_info.num_workers
        self.num_classes = cfg.data_info.num_classes
        cfg.input_resolution = (128,128)
        print("cfg.input_resolution:",cfg.input_resolution)
        data_dir = '../../../data'

        self.train_preprocessor = transforms.Compose([transforms.RandomCrop(128,128),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        self.trainset = DatasetFromFolder(data_dir + '/DIV2K/DIV2K_train_HR', transform=self.train_preprocessor)     

        self.trainloader = torch.utils.data.DataLoader(dataset=self.trainset, batch_size=self.batch_size,shuffle=True,
                                                       num_workers=self.num_workers,pin_memory=True,drop_last=False)
                                                       
class DIV2Kdataset_test():
    def __init__(self,cfg: DictConfig):
        self.test_preprocessor = transforms.Compose([transforms.ToTensor()])
        
        data_dir = '../../../data'

        self.testset = Datasets(data_dir + '/DIV2K/DIV2K_valid_HR')
        self.testloader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

class Kodakdataset_test():
    def __init__(self,cfg: DictConfig):
        self.test_preprocessor = transforms.Compose([transforms.ToTensor()])
        
        data_dir = '../../../data'

        self.testset = Datasets(data_dir + '/kodak')
        self.testloader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)            

class Flickr30kdataset_train():
    def __init__(self,cfg: DictConfig):
        self.data_name = cfg.data_info.data_name
        self.batch_size = cfg.data_info.batch_size          # per-GPU batch size (important!)
        self.num_workers = cfg.data_info.num_workers
        self.num_classes = cfg.data_info.num_classes
        cfg.total_max_epoch = 80
        print("cfg.input_resolution:", cfg.input_resolution)

        self.train_preprocessor = transforms.Compose([transforms.RandomResizedCrop((256,256)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        data_dir = '../../../data'

        self.trainset = DatasetFromFolder(
            data_dir + '/flickr30k/flickr30k-images',
            transform=self.train_preprocessor
        )

        # ---- DDP: use DistributedSampler for TRAIN ----
        self.train_sampler = None
        if cfg.use_ddp and dist.is_available():
            # sampler handles shuffling; do NOT use shuffle=True in DataLoader
            self.train_sampler = DistributedSampler(
                self.trainset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True,  # recommended for stable per-rank batch counts
            )

        self.trainloader = DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),     # True only for non-DDP
            sampler=self.train_sampler,               # None if non-DDP
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,                           # for training, usually True
            persistent_workers=(self.num_workers > 0)
        )

    def set_epoch(self, epoch: int):
        """Call this at each epoch start when using DDP."""
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
            

##https://github.com/leftthomas/SRGAN/blob/master/data_utils.py
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.transform = transform


    def __getitem__(self, index):
           
        image = self.transform(Image.open(self.image_filenames[index]))
        label = 0
        return image, label

    def __len__(self):
        return len(self.image_filenames)


##https://github.com/semcomm/SwinJSCC/blob/main/data/datasets.py
class Datasets(Dataset):
    def __init__(self, dataset_dir):
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        


    def __getitem__(self, item):
        image = Image.open(self.image_filenames[item]).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        img = self.transform(image)
        label = 0
        return img, label
    def __len__(self):
        return len(self.image_filenames)























  
