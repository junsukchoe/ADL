## ADL: Attention-based Dropout Layer for Weakly Supervised Object Localization

Tensorpack implementation of Attention-Dropout Layer for Weakly Supservised Object Localization.  

Our implementation is based on these repositories:
- [Tensorpack Saliency Example](https://github.com/tensorpack/tensorpack/tree/master/examples/Saliency)
- [Tensorpack ImageNet Example](https://github.com/tensorpack/tensorpack/tree/master/examples/ImageNetModels)

Imagenet Pre-trained models can be downloaded here:
- [Tensorpack Models](http://models.tensorpack.com/)

## Getting Started
### Requirements
- Python 3.3+
- Python bindings for OpenCV.
- Tensorflow (≥ 1.12, < 2)
- Tensorpack (= 0.9.0.1)

### Train & Test Examples
- CUB-200-2011
```
python CAM-VGG.py --gpu 0 --data /notebooks/dataset/CUB200/ --cub --base-lr 0.01 --logdir VGGGAP_CUB --load VGG --batch 32 --attdrop 3 4 53 --threshold 0.80 --keep_prob 0.25
```

- ImageNet
```
python CAM-VGG.py --gpu 0 --data /notebooks/dataset/ILSVRC2012/ --imagenet --base-lr 0.01 --logdir VGGGAP_ImageNet --load VGG --batch 256 --attdrop 3 4 53 --threshold 0.80 --keep_prob 0.25
```

## Coming Soon
* Detailed instructions
