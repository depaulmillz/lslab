#!/bin/bash

docker build -t lslabtest .
docker run --name lslabtest_1 --gpus all -it lslabtest
docker rm lslabtest_1
