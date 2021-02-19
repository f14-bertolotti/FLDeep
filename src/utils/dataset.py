from utils.image import Image
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
               torch.stack([d[2] for d in data]),\
               torch.stack([d[3] for d in data]),\
               torch.stack([d[4] for d in data]),\
               torch.stack([d[5] for d in data])

    @staticmethod
    def np_normalize_annotations(annotations, osize, bbox):
        return numpy.stack([(annotations[:,0] - bbox[0][0])/osize[1]-.5, (annotations[:,1] - bbox[0][1])/osize[0]-.5],axis=1)

    @staticmethod
    def np_recover_annotations(annotations, osize, bbox):
         return numpy.stack([(annotations[:,0]+.5)*osize[1]+bbox[0][0], (annotations[:,1]+.5)*osize[0]+bbox[0][1]],axis=1)

    @staticmethod
    def th_normalize_annotations(annotations, osize, bbox):
        return torch.stack([(annotations[:,:,0] - bbox[:,0,0].unsqueeze(-1))/osize[:,1].unsqueeze(-1)-.5, (annotations[:,:,1] - bbox[:,0,1].unsqueeze(-1))/osize[:,0].unsqueeze(-1)-.5],dim=2)

    @staticmethod
    def th_recover_annotations(annotations, osize, bbox):
        return torch.stack([(annotations[:,:,0]+.5)*osize[:,1].unsqueeze(-1)+bbox[:,0,0].unsqueeze(-1), (annotations[:,:,1]+.5)*osize[:,0].unsqueeze(-1)+bbox[:,0,1].unsqueeze(-1)],dim=2)


       

    def preprocess(self, image_path, annotations):
        image = cv2.cvtColor(cv2.imread(os.path.join(self.dataset_path, image_path)),cv2.COLOR_BGR2RGB)
        bbox  = ((annotations[0,0],annotations[0,1]),(annotations[1,0],annotations[1,1]))      
        image = image[int(round(bbox[0][1])):int(round(bbox[1][1])),int(round(bbox[0][0])):int(round(bbox[1][0]))] 
        osize = (image.shape[0], image.shape[1])                                              
        nsize = (self.configuration.image_size[0], self.configuration.image_size[1])       
        image = cv2.resize(image,(nsize[0],nsize[1]),interpolation=cv2.INTER_NEAREST)
        inter_corner_distance = numpy.linalg.norm(annotations[36,:] - annotations[45,:])
        true  = annotations
        gold  = Dataset.np_normalize_annotations(annotations, osize, bbox)
        
        data = torch.tensor(image/255             , dtype=torch.float, requires_grad=False)
        true = torch.tensor(annotations[7:]       , dtype=torch.float, requires_grad=False)
        gold = torch.tensor(gold[7:]              , dtype=torch.float, requires_grad=False)
        icds = torch.tensor(inter_corner_distance , dtype=torch.float, requires_grad=False)
        size = torch.tensor(osize                 , dtype=torch.float, requires_grad=False)
        bbox = torch.tensor(bbox                  , dtype=torch.float, requires_grad=False)
        return data, gold, true, icds, size, bbox

    def sample(self):
        return self.preprocess(*random.choice(self.data))

    def __getitem__(self,i):
        return self.preprocess(*self.data[i])

    def __iter__(self):
        return iter(map(lambda x:self.preprocess(*x), self.data))

    def __len__(self):
        return len(self.data)
