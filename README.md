## ADL: Attention-based Dropout Layer for Weakly Supervised Object Localization

Implementation of Attention-Dropout Layer for Weakly Supservised Object Localization.  


**[Junsuk Choe](https://junsukchoe.github.io/), [Hyunjung Shim](https://sites.google.com/site/katehyunjungshim/)**  
Vision and Learning Lab., Yonsei University.


[Attention-based Dropout Layer for Weakly Supervised Object Localization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Choe_Attention-Based_Dropout_Layer_for_Weakly_Supervised_Object_Localization_CVPR_2019_paper.pdf), CVPR'19 (Oral)


Weakly Supervised Object Localization (WSOL) techniques learn the object location only using image-level labels, without location annotations. A common limitation for these techniques is that they cover only the most discriminative part of the object, not the entire object. To address this problem, we propose an Attention-based Dropout Layer (ADL), which utilizes the self-attention mechanism to process the feature maps of the model. The proposed method is composed of two key components: 1) hiding the most discriminative part from the model for capturing the integral extent of object, and 2) highlighting the informative region for improving the recognition power of the model. Based on extensive experiments, we demonstrate that the proposed method is effective to improve the accuracy of WSOL, achieving a new state-of-the-art localization accuracy in CUB-200-2011 dataset. We also show that the proposed method is much more efficient in terms of both parameter and computation overheads than existing techniques.

<img width="1000" alt="teaser" src="img/overview.png">  

## Updates
**21 July, 2019**: Pytorch implementation released  
**23 June, 2019**: Initial upload

## Citation
If this work is useful for your research, please cite with:
```
@inproceedings{choe2019attention,
  title={Attention-Based Dropout Layer for Weakly Supervised Object Localization},
  author={Choe, Junsuk and Shim, Hyunjung},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2219--2228},
  year={2019}
}
```
