from utils.augmentations import Augment

import torchvision
import random
import torch
import numpy
import cv2
import os

random.seed(14)

class Dataset:
    def __init__(self, path, configuration):
        self.dataset_path    = path
        self.configuration   = configuration
        self.annotation_path = os.path.join(self.dataset_path,
                                            list(filter(lambda x:x.endswith(".txt"), 
                                                        os.listdir(self.dataset_path)))[0])
        self.data = list(map(lambda x:[x.split(" ",1)[0],numpy.array(list(map(float, x.split(" ",1)[1].split(" ")))).reshape(-1,2)], 
                             open(self.annotation_path,"r").read().split(" \n")[:-1]))

    @staticmethod
    def collate_fn(data):
        return torch.stack([d[0] for d in data]),\
               torch.stack([d[1] for d in data]),\
               torch.stack([d[2] for d in data])

    def annotation_forward(self,gold):
        gold[1:,0] -= gold[0,0]
        gold[1:,1] -= gold[0,1]
        gold[2:,0] /= gold[1,0] / self.configuration.image_size[0]
        gold[2:,1] /= gold[1,1] / self.configuration.image_size[1]
        gold[3:,0] -= self.configuration.image_size[0] / 2
        gold[3:,1] -= self.configuration.image_size[1] / 2
        return gold

    def annotation_backward(self,gold):
        gold[3:,0] += self.configuration.image_size[0] / 2
        gold[3:,1] += self.configuration.image_size[1] / 2
        gold[2:,1] *= gold[1,1] / self.configuration.image_size[1]
        gold[2:,0] *= gold[1,0] / self.configuration.image_size[0]
        gold[1:,1] += gold[0,1]
        gold[1:,0] += gold[0,0]
        return gold

    def preprocess(self, image_path, annotations):
        image = torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(self.dataset_path, image_path)),cv2.COLOR_BGR2RGB), dtype=torch.float, requires_grad=False).transpose(0,2).transpose(1,2)/256
        true  = torch.tensor(annotations, dtype=torch.float, requires_grad=False)
        
        true[0,0],true[0,1],true[1,0],true[1,1] = max(true[0,0],0),max(true[0,1],0),max(true[1,0],0),max(true[1,1],0)
        face  = torchvision.transforms.functional.crop(image,int(true[0,1].round()),int(true[0,0].round()),  # get face
                                                             int((true[1,1]-true[0,1]).round()),             #
                                                             int((true[1,0]-true[0,0]).round()))             #
        
        face = torchvision.transforms.functional.resize(face,self.configuration.image_size)

        gold = self.annotation_forward(true.clone())
        return face, gold, true

    def sample(self):
        return self.preprocess(*random.choice(self.data))

    def __getitem__(self,i):
        return self.preprocess(*self.data[i])

    def __iter__(self):
        return iter(map(lambda x:self.preprocess(*x), self.data))

    def __len__(self):
        return len(self.data)
