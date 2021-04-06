from utils.dataset import Dataset
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision

class Image:
    no_saved = 0

    @staticmethod
    def show(configuration, image, annotations, annotations_gold=numpy.array([]), **kwargs):
        fig,ax = plt.subplots(1,1)
        ax.imshow(image,**kwargs)
        list(map(lambda x:ax.scatter(x[:,0]+configuration.image_size[0]//2,x[:,1]+configuration.image_size[1]//2,s=4, color="black"), annotations))
        list(map(lambda x:ax.scatter(x[:,0]+configuration.image_size[0]//2,x[:,1]+configuration.image_size[1]//2,s=3, color= "lime"), annotations))
        fig.savefig(configuration.epoch_log_path+str(Image.no_saved))
        plt.close()
        Image.no_saved += 1

    @staticmethod
    def showall(images, preds, golds, configuration, dataset=None):
        for image, p, g in zip(images, preds, golds):
            Image.show(configuration, image.transpose(0,2).transpose(0,1).cpu().numpy(),
                       p[7:].unsqueeze(0).cpu().numpy(),
                       g[7:].unsqueeze(0).cpu().numpy())
        

