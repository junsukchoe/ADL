#!/bin/bash

wget -nc -P dataset/ http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar xvf dataset/CUB_200_2011.tgz -C dataset/
mv dataset/CUB_200_2011/ dataset/CUB/