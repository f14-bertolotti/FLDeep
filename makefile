
UID := $(shell id -u)
THISFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
THISDIR  := $(dir $(THISFILE))

.Dockerfile: Dockerfile
	docker build . -t f14:FLDeep -f Dockerfile
	touch .Dockerfile

all: download train test

download:
	rm -rf ${THISDIR}data/*
	docker run --rm \
		--user $(UID) \
		--mount type=bind,src=${THISDIR},dst=/FacialLandmarkDetection/ \
		-it f14:FLDeep python3 download.py

train-gpu: .Dockerfile
	docker run --rm --gpus all \
			--user $(UID) \
			--env MPLCONFIGDIR=/mpl \
			--shm-size=2G \
	    	--mount type=bind,src=${THISDIR},dst=/FacialLandmarkDetection/ \
	    	-it f14:FLDeep python3 train.py

test-gpu: .Dockerfile
	docker run --rm --gpus all \
			--user $(UID) \
			--env MPLCONFIGDIR=/mpl \
			--shm-size=2G \
	    	--mount type=bind,src=${THISDIR},dst=/FacialLandmarkDetection/ \
	    	-it f14:FLDeep python3 test.py

train-cpu: .Dockerfile
	docker run --rm \
			--user $(UID) \
			--env MPLCONFIGDIR=/mpl \
			--shm-size=2G \
	    	--mount type=bind,src=${THISDIR},dst=/FacialLandmarkDetection/ \
	    	-it f14:FLDeep python3 train.py

test-cpu: .Dockerfile
	docker run --rm \
			--user $(UID) \
			--env MPLCONFIGDIR=/mpl \
			--shm-size=2G \
	    	--mount type=bind,src=${THISDIR},dst=/FacialLandmarkDetection/ \
	    	-it f14:FLDeep python3 test.py

