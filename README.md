# SLNet
This is an unofficial PyTorch implementation of [Lightweight Semi-Supervised Scene Classification Network for Remote Sensing Images]


## dataset

train:test=8:2
UC-Merced         Labeled 1  2  5  10     
AID               Labeled 1  2  5  10
NWPU-RESISC45     Labeled 1  2  5  10


## Usage

### Train
Train the model by 21 labeled data of ucm dataset:
```
python train.py --dataset ucm --num-labeled 21 --arch edgenext --batch-size 8 --lr 0.00001 --expand-labels --seed 7 --out results/ucm@21.7
```

Train the model by 30 labeled data of ucm dataset:
```
python train.py --dataset aid --num-labeled 30 --arch edgenext --batch-size 8 --lr 0.0001 --expand-labels --seed 7 --out results/aid@30.7
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
- [Official pytorch implementation of FixMatch](https://github.com/kekmodel/FixMatch-pytorch.git))
- [Unofficial PyTorch implementation of edgenext](https://github.com/mmaaz60/EdgeNeXt.git)
- [Unofficial PyTorch Reimplementation of adl](https://github.com/junsukchoe/ADL.git)
- [PyTorch image models](https://github.com/rwightman/pytorch-image-models)


