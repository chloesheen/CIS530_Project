#!/bin/bash
python run_bert.py \
  --model-type bert \
  --model-name bert-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type bert \
  --model-name bert-base-uncased \
  --max-length 512

python run_bert.py \
  --model-type distilbert \
  --model-name distilbert-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type camembert \
  --model-name camembert-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type albert \
  --model-name albert-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type flaubert \
  --model-name flaubert-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type xlnet \
  --model-name xlnet-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type xlm \
  --model-name xlm-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type roberta \
  --model-name roberta-base-uncased \
  --max-length 128

python run_bert.py \
  --model-type xlmroberta \
  --model-name xlmroberta-base-uncased \
  --max-length 128
