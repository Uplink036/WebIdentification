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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
export_dir=${EXPORT_DIR:-"$repo_root/fetched_data"}

data_volume_name="webidentification_fetched_data"
volume_arg="$data_volume_name:/app/CV_WebIdentification"

docker run --rm \
    --net=host \
    --cpus="$limited_cpus" \
    -v "$volume_arg" \
    --env URI=bolt://localhost:7687 \
    --env USERNAME=neo4j \
    --env PASSWORD=password \
    webidentification_data_fetcher:latest

docker run --rm -v "$data_volume_name:/data:ro" alpine \
    tar -C /data -cf - . | tar -C "$export_dir" -xf -
