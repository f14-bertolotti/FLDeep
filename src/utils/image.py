from utils.dataset import Dataset
import matplotlib.pyplot as plt
import numpy
import torch

class Image:
    no_saved = 0
    @staticmethod
    def show(configuration, image, annotations, annotations_gold=numpy.array([]), **kwargs):
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(image,**kwargs)
        list(map(lambda x:ax[1].scatter(x[:,0],x[:,1],s=5), annotations_gold))
        list(map(lambda x:ax[1].scatter(x[:,0],x[:,1],s=5), annotations))
        ax[1].set_ylim(ax[1].get_ylim()[::-1])
        fig.savefig(configuration.epoch_log_path+str(Image.no_saved))
        plt.close()
        Image.no_saved += 1


    @staticmethod
    def showall(images, preds, golds, configuration, dataset=None):
        for image, p, g in zip(images, preds, golds):
            if dataset: p = dataset.annotation_backward(p)
            Image.show(configuration, image.transpose(0,2).transpose(0,1).cpu().numpy(),
                       p[7:].unsqueeze(0).cpu().numpy(),
                       g[7:].unsqueeze(0).cpu().numpy())
        

