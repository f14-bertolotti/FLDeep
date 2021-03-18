
NAME:= tmp
UID := $(shell id -u)
GID := $(shell id -g)


.Dockerfile: Dockerfile
	sudo docker build . -t f14:FLDeep -f Dockerfile
	touch .Dockerfile

docker-run: .Dockerfile
	xhost +local:root
	sudo docker run --runtime=nvidia --rm --env DISPLAY=$(DISPLAY) \
	    	--shm-size 2G \
	    	--net=host \
	    	--hostname=atavel \
	    	--mount type=bind,src=/home/f14/.cache/torch/hub/checkpoints/,dst=/root/.cache/torch/hub/checkpoints/ \
	    	--mount type=bind,src=/home/f14/gits/datasets/300W,dst=/FacialLandmarkDetection/data/300W \
	    	--mount type=bind,src=/home/f14/gits/models/FLDeep/$(NAME),dst=/FacialLandmarkDetection/model \
	    	--mount type=bind,src=/home/f14/gits/FacialLandmarkDetection/FLDeep/,dst=/FacialLandmarkDetection/ \
	    	--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix:rw \
	    	-it f14:FLDeep

train: .Dockerfile
	xhost +local:root
	sudo docker run --runtime=nvidia --rm --env DISPLAY=$(DISPLAY) \
	    	--shm-size 2G \
	    	--net=host \
	    	--hostname=atavel \
	    	--mount type=bind,src=/home/f14/.cache/torch/hub/checkpoints/,dst=/root/.cache/torch/hub/checkpoints/ \
	    	--mount type=bind,src=/home/f14/gits/datasets/300W,dst=/FacialLandmarkDetection/data/300W \
	    	--mount type=bind,src=/home/f14/gits/models/FLDeep/$(NAME),dst=/FacialLandmarkDetection/model \
	    	--mount type=bind,src=/home/f14/gits/FacialLandmarkDetection/FLDeep/,dst=/FacialLandmarkDetection/ \
	    	--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix:rw \
	    	-it f14:FLDeep \
		python3 train.py


test: .Dockerfile
	xhost +local:root
	sudo docker run --runtime=nvidia --rm --env DISPLAY=$(DISPLAY) \
	    	--shm-size 2G \
	    	--net=host \
	    	--hostname=atavel \
	    	--mount type=bind,src=/home/f14/.cache/torch/hub/checkpoints/,dst=/root/.cache/torch/hub/checkpoints/ \
	    	--mount type=bind,src=/home/f14/gits/datasets/300W,dst=/FacialLandmarkDetection/data/300W \
	    	--mount type=bind,src=/home/f14/gits/models/FLDeep/$(NAME),dst=/FacialLandmarkDetection/model \
	    	--mount type=bind,src=/home/f14/gits/FacialLandmarkDetection/FLDeep/,dst=/FacialLandmarkDetection/ \
	    	--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix:rw \
	    	-it f14:FLDeep \
		python3 test.py


