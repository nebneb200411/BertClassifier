#!/bin/bash

mkdir checkpoints
cd checkpoints

echo "get bert models"
git clone https://huggingface.co/nlp-waseda/roberta-base-japanese
