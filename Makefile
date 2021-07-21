PROJECT ?= signate_jpx
DATADIR ?= ${PWD}/data
WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= ${PROJECT}:latest
SUBMIT_NAME ?= tutorial
LOCAL_UID ?= 1000
LOCAL_GID ?= 1000

DOCKER_OPTS := \
			--name ${PROJECT} \
			--rm -it \
			-v ${PWD}:${WORKSPACE} \
			-w ${WORKSPACE} \
			--network=host \
			-e LOCAL_UID=${LOCAL_UID} \
			-e LOCAL_GID=${LOCAL_GID}

docker-build:
	docker build -f docker/Dockerfile -t ${DOCKER_IMAGE} .

docker-start-interactive: docker-build
	docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-start-jupyter: docker-build
	docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "jupyter lab --port=8888 --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''"

docker-run: docker-build
	docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "${COMMAND}"

docker-create-submit-files: docker-build
	docker run ${DOCKER_OPTS} --memory=8gb ${DOCKER_IMAGE} \
		bash -c "python tools/check_submit_files.py ${SUBMIT_NAME}"

docker-validate-submit-files: docker-build
	docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "python tools/check_submit_files.py ${SUBMIT_NAME} --validate"