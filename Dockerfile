FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install wget matplotlib opencv-python einops
#COPY . /FacialLandmarkDetection/
WORKDIR /FacialLandmarkDetection/

