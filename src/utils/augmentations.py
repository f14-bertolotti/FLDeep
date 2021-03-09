
from math import sin, cos, radians
import torchvision
import torch

import matplotlib.pyplot as plt
class Augment:
    
    def __init__(self, configuration):
        self.configuration = configuration

    def random_rotate(self, image, annot):

        angle = torchvision.transforms.RandomRotation.get_params([-self.configuration.augmentation_angle,self.configuration.augmentation_angle])
        pivot = torch.tensor([annot[0,0] + (annot[1,0] - annot[0,0])/2, annot[0,1] + (annot[1,1] - annot[0,1])/2])
        image = torchvision.transforms.functional.rotate(image,angle,center=pivot.tolist())
        angle = radians(-angle)
        rmatr = torch.tensor([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])
        annot[2:] = torch.einsum("ij,kj -> ki",rmatr,annot[2:]-pivot)+pivot

        return image, annot

    def random_flip(self, image, annot):
        
        if torch.rand(1) < self.configuration.augmentation_flipprob:
            pivot = torch.tensor([image.size(2),image.size(1)])/2
            image = torchvision.transforms.functional.hflip(image) 
            annot[:,0] = ((annot[:]-pivot)*-1+pivot)[:,0]
            annot[0,0],annot[1,0] = annot[1,0].item(),annot[0,0].item()

        return image, annot

