FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install wget matplotlib opencv-python einops
#COPY . /FacialLandmarkDetection/ 
WORKDIR /FacialLandmarkDetection/src

