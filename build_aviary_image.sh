#!/bin/bash
set -euxo pipefail

git diff-index --quiet HEAD --
GIT_COMMIT=`git rev-parse HEAD`

DOCKER_REPO="${AVIARY_DOCKER_REPO:-anyscale/aviary}"
VERSION="0.1.1"
DOCKER_TAG="$DOCKER_REPO:$VERSION-$GIT_COMMIT"
DOCKER_FILE="${AVIARY_DOCKER_FILE:-deploy/ray/Dockerfile}"

./build_aviary_wheel.sh

sudo docker build . -f $DOCKER_FILE -t $DOCKER_TAG -t $DOCKER_REPO:latest
sudo docker push -a "$DOCKER_REPO"
