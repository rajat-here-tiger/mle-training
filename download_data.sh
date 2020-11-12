#!/bin/sh
set -e
mkdir -p data/raw
wget https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz -O data/raw/housing.tar.gz
tar -xzf data/raw/housing.tar.gz --directory data/raw
rm data/raw/housing.tar.gz
