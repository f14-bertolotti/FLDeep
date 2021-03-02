from utils.dataset import Dataset
import torch

class Metrics:
    
    @staticmethod
    def normalized_mean_error(pred, true, dataset):
        error = 0
        for i,(p,t) in enumerate([(dataset.annotation_backward(p)[7:],t[7:]) for p,t in zip(pred.clone(),true)],1):
            d = torch.linalg.norm(t[36,:]-t[45,:])
            error += torch.linalg.norm(t-p,dim=-1).mean()/d
        return error * 100 / i



