from utils.configuration import Configuration
from utils.loggers       import DefaultLogger
from utils.model         import     SaveModel
from utils.metrics       import       Metrics
from utils.dataset       import       Dataset
from utils.image         import         Image
from pathlib             import          Path

import logging
import   torch
import    json
import     sys

configuration = Configuration("./configuration.json")
torch.manual_seed(configuration.random_seed)

stdout_logger = DefaultLogger.get_logger("STDOUT", stream=sys.stdout                , formatter=configuration.logging_format) if configuration.verbose        else DefaultLogger.get_fake_logger()
step_logger   = DefaultLogger.get_logger("STEP"  , file=configuration.step_log_path , formatter=configuration.logging_format) if configuration.step_log_path  else DefaultLogger.get_fake_logger()
epoch_logger  = DefaultLogger.get_logger("EPOCH" , file=configuration.epoch_log_path, formatter=configuration.logging_format) if configuration.epoch_log_path else DefaultLogger.get_fake_logger()


def evalueate(dataset,model):
    model.eval()
    loss = 0
    nme  = 0
    with torch.no_grad():
        for i,(data, gold, true) in enumerate(dataset,1):
            print("step:{}/{}, best:{: >3.5f}, nme:{: >3.5f}\r".format(i,len(dataset),loss,nme),end="")
            data = data.to(configuration.device)
            gold = gold.to(configuration.device)
            true = true.to(configuration.device)

            pred = model(data,gold)
            nme  = nme +(Metrics.normalized_mean_error(pred,true,valid)-nme)/i
            loss = loss+(model.loss(pred[:,7:],gold[:,7:]).item()-loss)/i
    model.train()
    return loss, nme


if __name__ == "__main__":

    stdout_logger.info(f"{__file__.upper()} STARTING")
    model = SaveModel(configuration) 
    configuration.load()
    model.train()
    stdout_logger.info(("resuming" if configuration.restore else "training") + f" epoch:{model.epoch}, best:{model.best}")
    epoch_logger.info(f"all_parameters {model.all_parameters}, trainable_parameters {model.trainable_parameters}")
    epoch_logger.info(f"{model.code}")

    train = Dataset(     configuration.train_path,configuration)
    valid = Dataset(configuration.validation_path,configuration)

    traindataset = torch.utils.data.DataLoader(train, batch_size=configuration.batch_size, collate_fn=Dataset.collate_fn, num_workers=1,prefetch_factor=1000, shuffle=True) 
    validdataset = torch.utils.data.DataLoader(valid, batch_size=configuration.batch_size, collate_fn=Dataset.collate_fn, num_workers=1,prefetch_factor=1000)

    for epoch in range(model.epoch, configuration.end_epoch):
        configuration.load()
        for i,(data, gold, _) in enumerate(traindataset,1):

            data = data.to(configuration.device)
            gold = gold.to(configuration.device)

            pred = model(data,gold)

            loss = model.loss(pred[:,7:],gold[:,7:])/configuration.mini_step_size
            loss.backward()

            if not i % configuration.steps_to_reload: configuration.load()
            if not i % configuration.mini_step_size: model.optimizer.step(); model.optimizer.zero_grad()

            with torch.no_grad():
                stdout_logger.info("epoch:{}, step:{}/{}, loss:{: >3.5f}".format(epoch,i,len(traindataset),loss.item()))
                step_logger  .info("epoch:{}, step:{}/{}, loss:{: >3.5f}".format(epoch,i,len(traindataset),loss.item()))

                if configuration.show_images: Image.showall(data,pred,gold,configuration)

       
        valid_loss,valid_nme = evalueate(validdataset, model)
        isbest = model.save_if_best(name="best", epoch=epoch+1, best=valid_nme)
        model.save(epoch=epoch+1, best=valid_nme)
        stdout_logger.info("EPOCH DONE. epoch:{}, loss:{: >3.5f}, nme:{: >3.5f}".format(epoch,valid_loss,valid_nme) + (" +" if isbest else ""))
        epoch_logger .info("EPOCH DONE. epoch:{}, loss:{: >3.5f}, nme:{: >3.5f}".format(epoch,valid_loss,valid_nme) + (" +" if isbest else ""))

    stdout_logger.info(f"{__file__.upper()} ENDING")
