#!/bin/sh

 # sudo ln -s /usr/bin/podman /usr/bin/docker
 # sudo ln -sf /run/user/1000/podman/podman.sock /var/run/docker.sock

mkdir ./artifacts || true
act \
  -v -r \
  --artifact-server-path ./artifacts \
  --container-architecture linux/amd64 \
  --container-daemon-socket "$DOCKER_HOST" --bind \
  --defaultbranch main \
  -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest \
  --artifact-server-path ./artifacts -j build

# -b - bind
# --rebuild - how to turn this off?
