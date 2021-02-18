import matplotlib.pyplot as plt
import numpy
import torch

class Image:
    @staticmethod
    def show(image, annotations, annotations_gold=numpy.array([])):
        fig,ax = plt.subplots(1,1)
        ax.imshow(image)
        list(map(lambda x:ax.scatter(x[:,0],x[:,1],s=5), annotations_gold))
        list(map(lambda x:ax.scatter(x[:,0],x[:,1],s=5), annotations))
        plt.show()

    @staticmethod
    def showall(images, preds, golds, configuration):
        for image, p, g in zip(images, preds, golds):
            print(torch.cat([p,g],-1),end="\n\n")
            p,g,image = p.detach().cpu(),g.detach().cpu(),image.detach().cpu()
            g[:,0] = (g[:,0] + .5) * configuration.image_size[0]
            g[:,1] = (g[:,1] + .5) * configuration.image_size[1]
            p[:,0] = (p[:,0] + .5) * configuration.image_size[0]
            p[:,1] = (p[:,1] + .5) * configuration.image_size[1]
            Image.show(image,p.unsqueeze(0),g.unsqueeze(0))
        

