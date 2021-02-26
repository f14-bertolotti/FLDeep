from utils.dataset import Dataset
import matplotlib.pyplot as plt
import numpy
import torch

class Image:
    @staticmethod
    def show(image, annotations, annotations_gold=numpy.array([]), **kwargs):
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(image,**kwargs)
        list(map(lambda x:ax[1].scatter(x[:,0],x[:,1],s=5), annotations_gold))
        list(map(lambda x:ax[1].scatter(x[:,0],x[:,1],s=5), annotations))
        ax[1].set_ylim(ax[1].get_ylim()[::-1])
        plt.show()


    @staticmethod
    def showall(images, preds, trues, configuration):
        pred = Dataset.unnoramlize(preds,trues) 
        for image, p, g in zip(images, preds, trues):
            Image.show(image.transpose(0,2).transpose(0,1),p.unsqueeze(0),g.unsqueeze(0))
        

