# FLDeep
This is my very simplistic take onto the task of facial landmark detection. 
I developed everything for two phd exams. I have put a bit of attention into reproducibility but I cannot guaranteed that everything will work flawlessly. 

# Prerequisites
Well, you just need ```Docker``` with the ``` nvidia-container-toolkit``` and ```make```.
Additionally, there is a single configuration file ```src/configuration.json```. Every option is commented. You should just need to edit this file to fit you needs. However, some things that you should know:
- the ```restore``` property should be always set to ```true``` during testing.
- the ```device``` property controls which device to use cpu/gpu. However docker will always try to use the gpu. If you don't have a GPU I don't know if it will work.
- the ```url``` property points to the dataset location. I do not mantain the dataset. Thus it can get unavailable. Anyway, you should be able to use any dataset as long as its structure matched the one of [300W].
- the ```train_path```, ```test_path``` and ```validation_path``` can be changed but they should point in a subdirectory of the main project directory. Otherwise, docker won't be able to find the dataset as it binds to main directory.
- One last thing. I have not test all the parameters exestensevly. Therefore, the probability of breaking something by changing a value is quite high. 

# Train
training a new model is fairly simple. set ```restore``` to ```false``` and run ```make train```.

# Testing
testing a trained model is also fairly simple. set ```restore``` to ```true``` and run ```make test``

[300W]: https://ibug.doc.ic.ac.uk/resources/300-W/

