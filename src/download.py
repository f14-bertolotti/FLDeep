from utils.configuration import Configuration

import logging
import zipfile
import wget
import sys
import os

configuration = Configuration("./configuration.json") 

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=configuration.logging_format)

if __name__ == "__main__":
    
    logging.info(f"{__file__.upper()} STARTING")
    
    if not os.path.isfile(configuration["zip_path"]):
        wget.download(configuration.url, out=configuration.zip_path)
    with zipfile.ZipFile(configuration.zip_path,"r") as zipref:
        zipref.extractall(os.path.dirname(configuration.zip_path))

    logging.info(f"{__file__.upper()} ENDED")

