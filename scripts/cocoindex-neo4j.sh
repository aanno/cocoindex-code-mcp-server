#!/bin/bash -x

podman compose -f ./cocoindex/dev/neo4j.yaml up -d
podman logs -f cocoindex-neo4j-neo4j-1
