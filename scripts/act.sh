#!/bin/sh

# mkdir ./artifacts || true
act \
  -v -r \
  --artifact-server-path ./artifacts \
  --container-architecture linux/amd64 \
  --defaultbranch main \
  -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest \
  --artifact-server-path ./artifacts -j build

# -b - bind
# --rebuild - how to turn this off?

