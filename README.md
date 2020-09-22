<div align="center">    
 
# DCGAN: Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks     

[![Paper](https://img.shields.io/badge/paper-arxiv.1511.06434-brightgreen)](https://arxiv.org/pdf/1511.06434)
[![Conference](https://img.shields.io/badge/Pytorch-DCGAN-orange)](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
[![Conference](https://img.shields.io/badge/Pytorch--Lightning-Homepage-blue)](https://pytorch-lightning.readthedocs.io/en/latest/)
  
</div>
 
## Description   
This project implement DCGAN with [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).In Pytorch DCGAN tutorial, they use the [Cele-A Faces dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for training, more details can be found in the [tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) and [Goodfellow's paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf).

## How to run   
First, install dependencies
```bash
# clone project   
git clone https://github.com/zwj2019/dcgan.git

# install requirements   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd dcgan

# run module, get more arguments in main.py and model/dcgan.py   
python main.py --dataroot=path/to/your/dataset

# sample, get more arguments in sample.py
python sample.py --weights=path/to/your/model/checkpoints
```
## TODO
- [x] Update requirements
- [x] Load weights from checkpoint
- [x] Complete sample.py