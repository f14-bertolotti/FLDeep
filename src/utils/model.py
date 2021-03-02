import pathlib
import random
import numpy
import types
import torch
import os

class SaveModel:

    def __init__(self, configuration):
        self.configuration = configuration
        self.code = pathlib.Path(self.configuration.class_path).read_text()
        self.epoch      = 0
        self.best       = float("inf")
        self.test_loss  = float("inf")
        if self.configuration.restore: self.load()
        else: self.model = SaveModel.import_code(self.code, "model").Model(configuration=configuration)

    @staticmethod
    def import_code(code, name):
        module = types.ModuleType(name)
        exec(code, module.__dict__)
        return module

    def __call__(self,*args,**kwargs):
        return self.model(*args,**kwargs)
        
    def __getattr__(self, name):
        return getattr(self.model, name)

    def load(self):
        checkpoint    = torch.load(self.configuration.model_path)
        configuration = checkpoint["configuration"]
        optimizersd   = checkpoint[  "optimizersd"]
        modelsd       = checkpoint[      "modelsd"] 
        code          = checkpoint[         "code"]
        best          = checkpoint[         "best"] if       "best" in checkpoint else self.best
        test_loss     = checkpoint[    "test_loss"] if  "test_loss" in checkpoint else  self.test_loss
        epoch         = checkpoint[        "epoch"] if      "epoch" in checkpoint else      self.epoch
        cudarng       = checkpoint[      "cudarng"]
        cpurng        = checkpoint[       "cpurng"]
        nprng         = checkpoint[        "nprng"]
        rngstate      = checkpoint[     "rngstate"]

        torch.cuda.set_rng_state(cudarng)
        torch.set_rng_state(cpurng)
        random.setstate(rngstate)
        numpy.random.set_state(nprng)

        self.code = code
        self.best = best
        self.test_loss  = test_loss
        self.epoch = epoch
        self.configuration.configuration.update(configuration)
        self.model = SaveModel.import_code(code, "model").Model(self.configuration)
        self.model.load_state_dict(modelsd)
        self.model.optimizer.load_state_dict(optimizersd)

    def save(self, name="", epoch=0, best=float("inf")):
        path = os.path.join(os.path.dirname(self.configuration.model_path), name + os.path.basename(self.configuration.model_path))
        torch.save({
            "configuration" : self.configuration.configuration,
              "optimizersd" : self.model.optimizer.state_dict(),
                  "modelsd" : self.model.state_dict(),
                     "code" : self.code,
                     "best" : self.best,
                  "cudarng" : torch.cuda.get_rng_state(),
                   "cpurng" : torch.get_rng_state(),
                    "nprng" : numpy.random.get_state(),
                 "rngstate" : random.getstate(),
                    "epoch" : epoch
            }, path)
    

    def save_if_best(self, name="", epoch=0, best=float("inf")):
        path = os.path.join(os.path.dirname(self.configuration.model_path), name + os.path.basename(self.configuration.model_path))
        if best < self.best:
            self.best = best
            torch.save({
                "configuration" : self.configuration.configuration,
                  "optimizersd" : self.model.optimizer.state_dict(),
                      "modelsd" : self.model.state_dict(),
                         "code" : self.code,
                         "best" : self.best,
                      "cudarng" : torch.cuda.get_rng_state(),
                       "cpurng" : torch.get_rng_state(),
                        "nprng" : numpy.random.get_state(),
                     "rngstate" : random.getstate(),
                        "epoch" : epoch
                }, path)
            return True
        return False
    

