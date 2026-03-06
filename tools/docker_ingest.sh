#!/bin/bash
set -euo pipefail

if command -v nproc >/dev/null 2>&1; then
    cpus=$(nproc)
elif command -v getconf >/dev/null 2>&1; then
    cpus=$(getconf _NPROCESSORS_ONLN)
else
    cpus=1
fi
cpu_limit=0.8
limited_cpus=$(awk -v c="$cpus" -v l="$cpu_limit" 'BEGIN { printf "%.1f", c * l }')

docker run --rm \
    --net=host \
    --cpus="$limited_cpus" \
    --env URI=bolt://localhost:7687 \
    --env USERNAME=neo4j \
    --env PASSWORD=password \
    webidentification_data_loader:latest