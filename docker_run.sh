#!/bin/bash

build=$1

if [ "$build" = "1" ]; then
    docker build -t speach_inference .
fi

docker run -it --rm \
    -v "$(pwd)/scripts/result.txt:/app/scripts/result.txt" \
    -v "$(pwd)/scripts/time.txt:/app/scripts/time.txt" \
    -v "$(pwd)/test:/app/test" \
    speach_inference