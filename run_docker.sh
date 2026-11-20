#!/bin/sh
# Usage: ./run_docker.sh <ids...>   (models/ and results/bench_input.npy mounted read-only; results written to ./results/bench via a writable mount)
docker build -t iot-bench .
mkdir -p results/bench
docker run --rm --cpus=1 --memory=1g \
  -v "$PWD/models:/app/models:ro" -v "$PWD/results/bench_input.npy:/app/results/bench_input.npy:ro" \
  -v "$PWD/results/bench:/app/results/bench" -v "$PWD/results:/app/results_out" \
  iot-bench python run_bench.py --docker docker "$@"
