#!/bin/bash

source .venv/bin/activate

cd ParlayANN/python

bash compile.sh

cd ../data_tools
make compute_groundtruth

cd ../..
uv run embeddings_to_fbin.py word2vec-google-news-300 50000