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
        for i,(data, gold, true, icds, size, bbox) in enumerate(dataset,1):
            print(f"step:{i}/{len(dataset)}, valid_loss:{loss}, nme:{nme}\r",end="")
            pred, gold = model(data,gold)
            nme  = nme +(Metrics.normalized_mean_error(pred.detach().to("cpu"),true,icds,size,bbox)-nme)/i
            loss = loss+(model.loss(pred,gold).item()-loss)/i
    model.train()
    return loss, nme


if __name__ == "__main__":

    stdout_logger.info(f"{__file__.upper()} STARTING")
    model = SaveModel(configuration) 
    model.train()
    stdout_logger.info(("resuming" if configuration.restore else "training") + f" epoch:{model.epoch}, valid_loss:{model.valid_loss}, test_loss:{model.test_loss}")

    traindataset = Dataset(     configuration.train_path,configuration)
    validdataset = Dataset(configuration.validation_path,configuration)

    traindataset = torch.utils.data.DataLoader(traindataset, batch_size=configuration.batch_size, collate_fn=Dataset.collate_fn, num_workers=1, prefetch_factor=10, shuffle=True)
    validdataset = torch.utils.data.DataLoader(validdataset, batch_size=configuration.batch_size, collate_fn=Dataset.collate_fn, num_workers=1, prefetch_factor=10)

    for epoch in range(model.epoch, configuration.end_epoch):
        for i,(data, gold, true, icds, size, bbox) in enumerate(traindataset,1):
            pred, gold = model(data,gold)
            loss = model.loss(pred,gold)/configuration.mini_step_size
            loss.backward()

            with torch.no_grad():
                nme = Metrics.normalized_mean_error(pred.to("cpu"),true,icds,size,bbox)
                stdout_logger.info(f"epoch:{epoch}, step:{i}/{len(traindataset)}, loss:{loss.item()}, nme:{nme}")
                step_logger  .info(f"epoch:{epoch}, step:{i}/{len(traindataset)}, loss:{loss.item()}, nme:{nme}")

            if configuration.show_images: Image.showall(data,pred,gold,configuration)

            if not i % configuration.steps_to_reload:
                configuration.load()

            if not i % configuration.mini_step_size:
                model.optimizer.step()
                model.optimizer.zero_grad()
        
        valid_loss,valid_nme = evalueate(validdataset, model)
        isbest = model.save_if_best(epoch=epoch+1, valid_loss=valid_loss)
        stdout_logger.info(f"EPOCH DONE. epoch:{epoch}, loss:{valid_loss}, nme:{nme}" + (" +" if isbest else ""))
        epoch_logger .info(f"EPOCH DONE. epoch:{epoch}, loss:{valid_loss}, nme:{nme}" + (" +" if isbest else ""))

    stdout_logger.info(f"{__file__.upper()} ENDING")
