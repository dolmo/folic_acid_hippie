#!/bin/bash
# Deploy Kubernetes job for the HIPPIE Folic Acid pipeline

#=============================================================================
# JOB IDENTIFICATION
#=============================================================================
export NAME="hippie-fa-pipeline"          
export JOB_PREFIX="prmanoj"               
export CONTAINER_NAME="hippie-image"      

#=============================================================================
# DOCKER IMAGE
#=============================================================================
# Upgraded to your partner's custom Docker image
export DOCKER_IMAGE="1r5agbgaofhieysdt9esr/folic-acid-hippie:latest"

#=============================================================================
# AWS/S3 CREDENTIALS (Sourced from your config.sh)
#=============================================================================
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"   
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
export S3_ENDPOINT="https://s3-west.nrp-nautilus.io"

#=============================================================================
# PATHS
#=============================================================================
export PROJECT_ROOT="/workspace"
export OUTPUT_DIR="/workspace/outputs"
export INPUT_S3_PATH="s3://braingeneersdev/prmanoj/folic_acid_t3/"
export S3_ENDPOINT_URL_RESULTS="s3://braingeneersdev/prmanoj/folic_acid_t3_results/pipeline_out/"

#=============================================================================
# RUN COMMAND
#=============================================================================
# The pod installs wget/git, downloads your script directly from GitHub, and runs it
export RUN_COMMAND="apt-get update && apt-get install -y wget git && wget -qO /workspace/run_pipeline_local.sh https://raw.githubusercontent.com/dolmo/folic_acid_hippie/main/run_pipeline_local.sh && bash /workspace/run_pipeline_local.sh"

#=============================================================================
# RESOURCE LIMITS
#=============================================================================
export MEMORY_LIMIT="32Gi"
export CPU_LIMIT="8"
export STORAGE_LIMIT="50Gi"
export GPU_LIMIT="1"

export MEMORY_REQUEST="16Gi"
export CPU_REQUEST="4"
export STORAGE_REQUEST="30Gi"

#=============================================================================
# VALIDATION & DEPLOY
#=============================================================================
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: AWS credentials not set! Run 'source config.sh' first."
    exit 1
fi

echo "Deploying Kubernetes job: ${JOB_PREFIX}-${NAME}"
envsubst < jobdefinition.yaml | kubectl apply -f -

echo "Job created successfully!"