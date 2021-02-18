from utils.configuration import Configuration
from utils.model         import     SaveModel
from utils.dataset       import       Dataset
from utils.image         import         Image
from pathlib             import          Path

import logging
import   torch
import    json
import     sys

configuration = Configuration("./configuration.json")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=configuration.logging_format)
torch.manual_seed(configuration.random_seed)

if __name__ == "__main__":

    logging.info(f"{__file__.upper()} STARTING")
    model = SaveModel(configuration) 
    model.eval()
    logging.info(f"testing epoch:{model.epoch}, valid_loss:{model.valid_loss}, test_loss:{model.test_loss}")
    
    testdataset = Dataset(configuration.test_path,configuration)
    
    testdataset = torch.utils.data.DataLoader(testdataset, batch_size=configuration.batch_size, collate_fn=Dataset.collate_fn, num_workers=1, prefetch_factor=10)
   
    with torch.no_grad():
        loss = None
        for i,(data,gold) in enumerate(testdataset,1):
            gold = gold.to(configuration.device)
            pred = torch.zeros(gold.shape).to(configuration.device)
            for j in range(68):
                print(f"step:{i}/{len(testdataset)}, pred:{j}/68, test_loss:{loss}\r",end="")
                pred[:,j] = model(data,gold)[0][:,j]
    
            configuration.load()
            if configuration.show_images: Image.showall(data,pred,gold,configuration)
    
            loss = loss+(model.loss(pred,gold).item()-loss)/i if loss else model.loss(pred,gold).item()
    
    logging.info(f"epoch:{model.epoch}, valid_loss:{model.valid_loss}, test_loss:{loss}")
    
    logging.info(f"{__file__.upper()} STARTING")
 

