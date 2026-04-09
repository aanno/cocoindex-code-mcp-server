#!/bin/bash -x

podman compose -f ./compose/postgres.yaml up -d
podman logs -f cocoindex-postgres_postgres_1
