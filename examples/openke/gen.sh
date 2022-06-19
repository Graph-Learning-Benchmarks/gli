#!/bin/bash
set -Eeuo pipefail
for dataset in FB13 FB15K237 NELL-995 WN11 WN18 WN18RR YAGO3-10
do
    echo + processing $dataset
    mkdir -p $dataset/raw
    cp ../../../OpenKE/benchmarks/$dataset/*.txt $dataset/raw
    cp -r FB15K/link_prediction $dataset
    cp FB15K/FB15K.ipynb $dataset/$dataset.ipynb
    cp FB15K/metadata.json $dataset/metadata.json
    cp FB15K/README.md $dataset/README.md
done