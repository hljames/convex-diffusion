#!/bin/bash

export ARGS="$@"   # not working as desired ..
kubectl create -f configs/run_sst_with_args.yaml
