
UID := $(shell id -u)
GID := $(shell id -g)

TORCHVISION_CACHE=/home/f14/.cache/torch/hub/checkpoints/
DATASET_DIRECTORY=/home/f14/heavy/datasets/300W
MODELS_DIRECTORY=/home/f14/heavy/models/FLDeep/tmp/

.Dockerfile: Dockerfile
	sudo docker build . -t f14:FLDeep -f Dockerfile
	touch .Dockerfile

docker-run: .Dockerfile
	xhost +local:root
	sudo docker run --runtime=nvidia --rm --env DISPLAY=$(DISPLAY) \
	    	--shm-size 2G \
	    	--net=host \
	    	--hostname=atavel \
	    	--mount type=bind,src=${TORCHVISION_CACHE},dst=/root/.cache/torch/hub/checkpoints/ \
	    	--mount type=bind,src=${DATASET_DIRECTORY},dst=/FacialLandmarkDetection/data/300W \
	    	--mount type=bind,src=${MODELS_DIRECTORY},dst=/FacialLandmarkDetection/model \
	    	--mount type=bind,src=${CURDIR},dst=/FacialLandmarkDetection/ \
	    	--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix:rw \
	    	-it f14:FLDeep

