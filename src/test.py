from utils.configuration import Configuration
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
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=configuration.logging_format)
torch.manual_seed(configuration.random_seed)

if __name__ == "__main__":

    logging.info(f"{__file__.upper()} STARTING")
    model = SaveModel(configuration) 
    configuration.load()
    model.eval()
    
    test = Dataset(configuration.test_path,configuration)
    testdataset = torch.utils.data.DataLoader(test, batch_size=configuration.batch_size, collate_fn=Dataset.collate_fn, num_workers=1, prefetch_factor=10)
   
    with torch.no_grad():
        loss = None
        nme  = None
        for i,(data, gold, true) in enumerate(testdataset,1):
            logging.info(f"batch:{i}/{len(testdataset)}, loss:{loss}, nme:{nme}")
            gold = gold.to(configuration.device)
            true = true.to(configuration.device)
            pred = torch.zeros(gold.shape).to(configuration.device)
            pred = model(data,gold)

            configuration.load()
            if configuration.show_images: Image.showall(data,pred.cpu(),true,configuration,dataset=test)
    
            loss = loss+(model.loss(pred[:,7:],gold[:,7:]).item()-loss)/i if loss else model.loss(pred[:,7:],gold[:,7:]).item()
            nme  = nme +(Metrics.normalized_mean_error(pred,true,test).item()-nme)/i if nme else Metrics.normalized_mean_error(pred,true,test).item()

    logging.info(f"average nme:{nme}")
    
    logging.info(f"{__file__.upper()} STARTING")
 

