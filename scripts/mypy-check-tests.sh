#!/bin/bash -x

mypy --config-file pyproject.toml --check-untyped-defs tests
