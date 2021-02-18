from utils.image import Image

import torchvision
import torch

class Model(torch.nn.Module):
    def __init__(self, configuration):
        super(Model,self).__init__()
    
        self.configuration = configuration

        self.resnet = torchvision.models.resnet18(pretrained=True, progress=True).to(configuration.device)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:8])


        self.sos          = torch.nn.Embedding(1,256).to(configuration.device)
        self.src_position = torch.nn.Embedding(512,256).to(configuration.device)
        self.tgt_position = torch.nn.Embedding(69,256).to(configuration.device)

        self.src_upscale  = torch.nn.Linear(49,256).to(configuration.device)
        self.tgt_upscale  = torch.nn.Linear(2,256).to(configuration.device)

        self.downscale = torch.nn.Linear(256,2).to(configuration.device)
        self.encoder   = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(256,nhead=4,dim_feedforward=1024,activation="relu",dropout=.1),num_layers=2).to(configuration.device)
        self.decoder   = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(256,nhead=4,dim_feedforward=1024,activation="relu",dropout=.1),num_layers=2).to(configuration.device)
        self.peek      = self.generate_square_subsequent_mask(69).to(configuration.device)

        self.loss         = torch.nn.L1Loss(reduction="sum")
        self.optimizer    = torch.optim.Adam(self.parameters(),lr=0.0001)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data, gold):
        with torch.no_grad():
            data = data.to(self.configuration.device)
            gold = gold.to(self.configuration.device)

            res = self.resnet(data.transpose(1,3))
            res = res.view(res.size(0),res.size(1),res.size(2)*res.size(3))

        
        srcs = self.src_upscale(res)
        tgts = torch.cat([self.sos.weight.unsqueeze(0).repeat(gold.size(0),1,1),self.tgt_upscale(gold)],1)

        srcs += self.src_position.weight.unsqueeze(0).repeat(gold.size(0),1,1)
        tgts += self.tgt_position.weight.unsqueeze(0).repeat(gold.size(0),1,1)

        mems = self.encoder(srcs.transpose(0,1))
        pred = self.decoder(memory=mems, tgt=tgts.transpose(0,1), tgt_mask=self.peek).transpose(0,1)
        ldmk = self.downscale(pred)

        return ldmk[:,:-1], gold

