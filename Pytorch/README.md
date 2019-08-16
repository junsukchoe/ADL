## ADL: Attention-based Dropout Layer for Weakly Supervised Object Localization

Pytorch implementation of Attention-Dropout Layer for Weakly Supervised Object Localization (in progress)    

Our implementation is based on these repositories:
- [Pytorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)

Imagenet Pre-trained model for ResNet-50 can be downloaded here:
- [pytorch-resnet](https://github.com/ruotianluo/pytorch-resnet/)

## Getting Started
### Requirements
- Python 3.3+
- Python bindings for OpenCV.
- Pytorch (≥ 1.1)
- TensorboardX
- OpenCV-python

### Train & Test Examples
- CUB-200-2011
```
git clone https://github.com/junsukchoe/ADL.git
cd ADL
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xvf CUB_200_2011.tgz
cd Pytorch
bash scripts/run1.sh
```

## Coming Soon
* ImageNet implementation
* Detailed instructions
