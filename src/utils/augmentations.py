
from math import sin, cos, radians
import torchvision
import einops
import torch

import matplotlib.pyplot as plt
class Augment:
    
    def __init__(self, configuration):
        self.configuration = configuration

    def random_rotate(self, images, annots):

        # sample rotations
        angles = torch.empty(images.size(0)).uniform_(-self.configuration.augmentation_angle, self.configuration.augmentation_angle).to(self.configuration.device)
        # apply rotations to images
        images = torch.stack([torchvision.transforms.functional.rotate(img,ang.item(),center=(self.configuration.image_size[0]/2,self.configuration.image_size[1]/2)) for img,ang in zip(images,angles)],0)
        # compute rotation matrices
        angles = einops.repeat(angles.deg2rad(),"i -> i h w ", h=2,w=2)*-1
        angles[:,0,0].cos_(); angles[:,0,1].sin_().mul_(-1); angles[:,1,0].sin_(); angles[:,1,1].cos_()
        # apply rotation matrices
        annots[:,2:] = torch.einsum("bij,bkj -> bki",angles,annots[:,2:])
 
        return images, annots

    def random_flip(self, images, annots):
 
        if torch.rand(1) < self.configuration.augmentation_flipprob:
            images.data = images.flip(-1) # flip image
            annots[:,2:,0].mul_(-1)       # flip annotations

        return images, annots

