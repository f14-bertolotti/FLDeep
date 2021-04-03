from utils.image import Image
import torch.nn.functional as fun

import torchvision
import einops
import torch
from utils.augmentations import Augment

class Model(torch.nn.Module):
    def __init__(self, configuration):
        super(Model,self).__init__()

        self.configuration = configuration

        self.resnet = torchvision.models.resnet101(pretrained=True, progress=True).to(configuration.device)
        self.resnet3 = torch.nn.Sequential(*list(self.resnet.children())[6:8])
        self.resnet2 = torch.nn.Sequential(*list(self.resnet.children())[5:6])
        self.resnet1 = torch.nn.Sequential(*list(self.resnet.children())[3:5])
        self.resnet0 = torch.nn.Sequential(*list(self.resnet.children())[0:3])

        self.reslinA0 = torch.nn.Linear(1024,2048).to(configuration.device)
        self.reslinA1 = torch.nn.Linear(1024,2048).to(configuration.device)
        self.reslinA2 = torch.nn.Linear(2048,2048).to(configuration.device)
        self.reslinA3 = torch.nn.Linear(2048,2048).to(configuration.device)
        self.reslinB0 = torch.nn.Linear(2048,512).to(configuration.device)
        self.reslinB1 = torch.nn.Linear(2048,512).to(configuration.device)
        self.reslinB2 = torch.nn.Linear(2048,512).to(configuration.device)
        self.reslinB3 = torch.nn.Linear(2048,512).to(configuration.device)
        self.dropout  = torch.nn.Dropout(.1)

        self.src_position = torch.nn.Embedding(441,512).to(configuration.device)

        self.downscale = torch.nn.Linear(512,2).to(configuration.device)
        self.encoder   = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(512,nhead=8,dim_feedforward=2048,activation="gelu",dropout=.1),num_layers=4).to(configuration.device)

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        self.loss         = torch.nn.L1Loss()
        self.optimizer    = torch.optim.Adam(self.parameters(),lr=0.0001)
        self.all_parameters = sum(p.numel() for p in self.parameters())
        self.trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.augment = Augment(self.configuration)

    def forward(self, data, gold):
        with torch.no_grad():
            data = data.to(self.configuration.device)
            gold = gold.to(self.configuration.device)
            
            if self.training and self.configuration.do_augmentation: 
                data,gold = self.augment.random_rotate(*self.augment.random_crop(*self.augment.random_jitter(data,gold)))

        res0 = self.resnet0(self.normalize(data))
        res1 = self.resnet1(res0)
        res2 = self.resnet2(res1)
        res3 = self.resnet3(res2)

        # divide intermediate representation in patches + linear
        res0 = self.reslinB0(self.dropout(fun.gelu(self.reslinA0(einops.rearrange(res0, "b c (h1 h2) (w1 w2) -> b (h2 w2) (c h1 w1)",h1=4,w1=4)))))
        res1 = self.reslinB1(self.dropout(fun.gelu(self.reslinA1(einops.rearrange(res1, "b c (h1 h2) (w1 w2) -> b (h2 w2) (c h1 w1)",h1=2,w1=2)))))
        res2 = self.reslinB2(self.dropout(fun.gelu(self.reslinA2(einops.rearrange(res2, "b c (h1 h2) (w1 w2) -> b (h2 w2) (c h1 w1)",h1=2,w1=2)))))
        res3 = self.reslinB3(self.dropout(fun.gelu(self.reslinA3(einops.rearrange(res3, "b c h w -> b (h w) c")))))

        # maxpool on channels
        res0 = einops.reduce(res0, "b (c1 c2) e -> b c1 e","max",c2=8)
        res1 = einops.reduce(res1, "b (c1 c2) e -> b c1 e","max",c2=4)
        res2 = einops.reduce(res2, "b (c1 c2) e -> b c1 e","max",c2=2)

        res = torch.cat([res0,res1,res2,res3],1)

        srcs = res + self.src_position.weight.unsqueeze(0).repeat(gold.size(0),1,1)
        mems = self.encoder(srcs.transpose(0,1)).transpose(0,1)
        ldmk = self.downscale(mems[:,:68])

        return torch.cat([gold[:,:7],ldmk],1)


