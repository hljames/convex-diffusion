#!/bin/bash

ids=("1b3wfroe")
for id in "${ids[@]}"; do
    echo "kubectl create $id"
    export WANDB_ID=$id
    # delete any previous kubectl pods
    # kubectl delete
    # Load template
    JOB_CONF=$(<configs/resume.yaml)
    # kubectl create -f configs/resume.yaml

# Substitute template vars and submit job
envsubst <<EOF | kubectl create -f -
$JOB_CONF
EOF

done