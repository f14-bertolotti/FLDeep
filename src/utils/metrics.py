from utils.dataset import Dataset
import torch

class Metrics:
    
    @staticmethod
    def normalized_mean_error(pred, true, icds, size, bbox):
        return (torch.linalg.norm(Dataset.th_recover_annotations(pred, size, bbox) - true,dim=-1).sum(-1) / icds).mean()



