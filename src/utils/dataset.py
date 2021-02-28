import torchvision
import random
import torch
import numpy
import cv2
import os

random.seed(14)

class Dataset:
    def __init__(self, path, configuration, docrop=True):
        self.docrop          = docrop
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
               torch.stack([d[2] for d in data]),\
               torch.stack([d[3] for d in data])

    @staticmethod
    def unnoramlize(pred,true):
        pred[:,:,0] *= true[:,1,0].unsqueeze(-1)
        pred[:,:,1] *= true[:,1,1].unsqueeze(-1)
        pred[:,:,0] += true[:,0,0].unsqueeze(-1)
        pred[:,:,1] += true[:,0,1].unsqueeze(-1)
        return pred

    def preprocess(self, image_path, annotations):
        #image = torchvision.io.read_image(os.path.join(self.dataset_path,image_path)) # get image 
        image = torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(self.dataset_path, image_path)),cv2.COLOR_BGR2RGB), dtype=torch.float, requires_grad=False).transpose(0,2).transpose(1,2)/255
        bimage = [[0,0],[image.shape[2],image.shape[1]]]                               # get shape

        bface = [[annotations[0,0],annotations[0,1]],[annotations[1,0],annotations[1,1]]]                                # get face bbox 
        face  = torchvision.transforms.functional.crop(image,int(round(annotations[0,1])),int(round(annotations[0,0])),  # get face
                                                             int(round(annotations[1,1]-annotations[0,1])),              #
                                                             int(round(annotations[1,0]-annotations[0,0])))              #
        
        i,j,h,w = torchvision.transforms.RandomResizedCrop.get_params(face,scale=(0.08, 1.0), ratio=(0.75, 4/3)) # get face crop params
        if not self.docrop: i,j,w,h = 0,0,int(round(bface[1][0]-bface[0][0])),int(round(bface[1][1]-bface[0][1]))
        bcrop   = [[j+bface[0][0],i+bface[0][1]],[j+bface[0][0]+w,i+bface[0][1]+h]]                              # get face crop bbox 
        crop    = torchvision.transforms.functional.crop(face,i,j,h,w)                                           # get face crop

        icd = numpy.linalg.norm(annotations[36,:] - annotations[45,:]) # compute normalization factor

        crop_resized = torchvision.transforms.functional.resize(crop, (224,224)) # resnet resolution

        gold = annotations.copy()
        gold[1:,0] -= annotations[0,0] # center annotations
        gold[1:,1] -= annotations[0,1] # center annotations
        gold[2:,0] /= annotations[1,0] # bbox normalized 
        gold[2:,1] /= annotations[1,1] # bbox normalized

        gold = torch.tensor(gold       , dtype=torch.float, requires_grad=False)
        true = torch.tensor(annotations, dtype=torch.float, requires_grad=False)
        icd  = torch.tensor(icd        , dtype=torch.float, requires_grad=False)
        data = crop_resized.float()
        

        return data, gold, true, icd

    def sample(self):
        return self.preprocess(*random.choice(self.data))

    def __getitem__(self,i):
        return self.preprocess(*self.data[i])

    def __iter__(self):
        return iter(map(lambda x:self.preprocess(*x), self.data))

    def __len__(self):
        return len(self.data)
