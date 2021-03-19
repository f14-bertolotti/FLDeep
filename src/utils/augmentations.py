
from math import sin, cos, radians
import torchvision
import einops
import torch

import matplotlib.pyplot as plt
class Augment:
    
    def __init__(self, configuration):
        self.configuration = configuration
        self.flip_rearrange = [0,1,2,3,4,5,6,]

    def random_rotate(self, images, annots):
        with torch.no_grad():
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

    def random_flip(self, images, annotpts):
        with torch.no_grad():
            flipmask = torch.rand(images.size(0)) < self.configuration.augmentation_flipprob
            images[flipmask] = images[flipmask].flip(-1)
            annotpts[flipmask,2:,0] = annotpts[flipmask,2:,0].mul(-1)
            annotpts[flipmask] = annotpts[flipmask][:,[0,1,2,3,4,5,6, # unchanged
                                                       23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7, # jaw line
                                                       33,32,31,30,29,28,27,26,25,24, # eye brows
                                                       34,35,36,37, # nose line
                                                       42,41,40,39,38, # nose
                                                       52,51,50,49,54,53, # right eye
                                                       46,45,44,43,48,47, # left eye
                                                       61,60,59,58,57,56,55, # upper upper lip
                                                       66,65,64,63,62, # lower lower lip
                                                       71,70,69,68,67, # upper lip
                                                       74,73,72],:] # lower lip
     


        return images, annotpts

