#!/bin/bash

cd ../environment
docker build . -t climateml:latest  # Build the docker image
docker tag climateml:latest salv4/climateml:latest  # Tag the image
docker push salv4/climateml:latest  # Push the image to dockerhub

# to push to nautilus gitlab:
# docker login gitlab-registry.nrp-nautilus.io
# docker build -t gitlab-registry.nrp-nautilus.io/salvarc/climate-ml .
# docker push gitlab-registry.nrp-nautilus.io/salvarc/climate-ml:latest