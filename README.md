# FLDeep
This is my very simplistic take on the task of facial landmark detection. 
I developed everything for two Ph.D. exams. I have put a bit of attention into reproducibility but I cannot guarantee that everything will work flawlessly. 

# Prerequisites
Well, you just need ```Docker``` with the ``` nvidia-container-toolkit``` and ```make```.
Additionally, there is a single configuration file ```src/configuration.json```. Every option is commented on. You should just need to edit this file to fit your needs. However, some things that you should know:
- The ```restore``` property should be always set to ```true``` during testing.
- The ```device``` property controls which device to use CPU/GPU. However, docker will always try to use the GPU. If you don't have a GPU I don't know if it will work.
- The ```url``` property points to the dataset location. I do not maintain the dataset. Thus it can get unavailable. Anyway, you should be able to use any dataset as long as its structure matched the one of [300W].
- The ```train_path```, ```test_path``` and ```validation_path``` can be changed but they should point in a subdirectory of the main project directory. Otherwise, docker won't be able to find the dataset as it binds to the main directory.
- The ```device``` property controls the usage of the GPU. It should be chosen accordingly with the targets from the makefile.
- One last thing. I have not tested all the parameters extensively. Therefore, the probability of breaking something by changing a value is quite high. 


# The architecture
The chosen architecture is a bit unconventional and it is inspired by the [Vision Transformer] with some twist.
Briefly, instead of taking patches only from the original image, the implemented Transformer takes patches from images obtained at various layers of a pretrained ResNet101.
You can find more details in this project [report].

# Train
training a new model is fairly simple. set ```restore``` to ```false``` and run ```make train-gpu``` or ```make train-cpu```.

# Testing
testing a trained model is also fairly simple. set ```restore``` to ```true``` and run ```make test-gpu``` or ```make test-cpu```.

# Results
Once trained you should be able to obtain very similar results:

| model     | NME(%) |
| --------- | ------ |
| [3DDE]    | 3.13   |
| [CNNCRF]  | 3.30   |
| [Adaloss] | 3.31   |
| [SAN-GT]  | 3.98   |
| this      | 4.26   |
| [CFSS]    | 5.76   |

As you can see our architecture can achieve results that are quite close to the state-of-the-art. Probably, with a bit more work and research, the performance could be pushed further below. But this is just speculation.

# Samples
Here, you have two examples of predictions from the test set of [300W].
![alt text](https://github.com/f14-bertolotti/FLDeep/blob/master/report/figs/prediction1.png?raw=true)
![alt text](https://github.com/f14-bertolotti/FLDeep/blob/master/report/figs/prediction2.png?raw=true)


[300W]: https://ibug.doc.ic.ac.uk/resources/300-W/
[Vision Transformer]: https://arxiv.org/abs/2010.11929
[ResNet101]: https://pytorch.org/hub/pytorch_vision_resnet/
[report]: https://github.com/f14-bertolotti/FLDeep/blob/main/report/main.pdf
[3DDE]: https://arxiv.org/pdf/1902.01831v2.pdf
[CNNCRF]: https://proceedings.neurips.cc/paper/2019/file/56352739f59643540a3a6e16985f62c7-Paper.pdf
[Adaloss]: https://arxiv.org/pdf/1908.01070v1.pdf
[SAN-GT]: https://arxiv.org/pdf/1803.04108v4.pdf
[CFSS]: https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhu_Face_Alignment_Across_CVPR_2016_paper.pdf
