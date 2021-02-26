from utils.dataset import Dataset
import torch

class Metrics:
    
    @staticmethod
    def normalized_mean_error(pred, true, icds):
        pred = Dataset.unnoramlize(pred,true)
        return (torch.linalg.norm(pred - true[:,7:],dim=-1).sum(-1) / icds).mean()



