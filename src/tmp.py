from utils.configuration import Configuration 
configuration = Configuration("./configuration.json") 
from utils.dataset       import       Dataset 
traindataset = Dataset(     configuration.train_path,configuration) 
import matplotlib.pyplot as plt 
image,fl,tr,icd,sz,bb = traindataset.sample() 
plt.scatter((fl[:,0]+.5)*224,(fl[:,1]+.5)*224,s=5) 
plt.imshow(image) 
plt.show()      
