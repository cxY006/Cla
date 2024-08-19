# SLNet
This is an unofficial PyTorch implementation of [Lightweight Semi-Supervised Scene Classification Network(SLNet) for Remote Sensing Images]


## dataset

UC-Merced
AID
NWPU-RESISC45


## Usage

### Train
Train the model by 21 labeled data of ucm dataset:

```
python train.py --dataset ucm --num-labeled 21 --arch edgenext --batch-size 8 --lr 0.0001 --expand-labels --seed 7 --out results/ucm@21.7
```

Train the model by 45 labeled data of nwpu dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --dataset nwpu --num-labeled 45 --arch edgenext --batch-size 8 --lr 0.0006 --wdecay 0.001 --expand-labels --seed 7 --out results/nwpu@45
```


## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)



## References
- [Official TensorFlow implementation of FixMatch](https://github.com/google-research/fixmatch)
- [Unofficial PyTorch implementation of MixMatch](https://github.com/YU1ut/MixMatch-pytorch)
- [Unofficial PyTorch Reimplementation of RandAugment](https://github.com/ildoonet/pytorch-randaugment)
- [PyTorch image models](https://github.com/rwightman/pytorch-image-models)


