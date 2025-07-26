#!/bin/bash -x

podman compose -f ./compose/qdrant.yaml up -d
podman logs -f cocoindex-qdrant-qdrant-1
