import argparse
import logging
import math
import random
import shutil
import time
import os
import util


from collections import OrderedDict
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset.data import DATASET_GETTERS
# from dataset.cifar import DATASET_GETTERS1
from utils import AverageMeter, accuracy
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from models.edge import model, edgenext
from timm.optim.sgdp import SGDP
from ptflops import get_model_complexity_info
import adabound


logger = logging.getLogger(__name__)
best_acc = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# iter_text = np.loadtxt("./data/iter.txt")
# loss_text = np.loadtxt("./data/loss.txt")

x = []
y = []



def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# def iou(pred, mask):
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)).sum(dim=(2, 3))
#     union = ((pred + mask)).sum(dim=(2, 3))
#     iou = 1 - (inter + 1) / (union - inter + 1)
#     return iou.mean()

#warm up--------------------
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
#warm up---------------------

def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='ucm', type=str,
                        choices=['cifar10', 'cifar100', 'ucm', 'nwpu'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=100,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='edgenext_base', type=str,  # pvt_v2_b0----pvt_tiny
                        choices=['wideresnet', 'resnext', 'pvt_tiny','pvt_small','pvt_large', 'vit_tiny_patch16_256', 'pvt_v2_b0','Semiformer', 'vit_base_patch16_224','pvt_v2_b3','pvt_v2_b5' , 'localvit_pvt' , 'edgenext_base'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2 ** 16, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=256, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.006, type=float,  # 0.01---0.001
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=3e-5, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1.4, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    #warmup-----------------------
    # parser.add_argument('--gpu', default=None, type=int,
    #                     help='GPU id to use.')
    # parser.add_argument('--thresh_warmup', type=stool, default=True)
    # parser.add_argument('--num_classes', type=int, default=21)
    #warmup------------------------
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    # parser.add_argument("--amp", action="store_true",
    #                     help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    # parser.add_argument("--opt_level", type=str, default="O1",
    #                     help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #                          "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--classifier_dropout', default=0.0, type=float)
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--finetune', default='',
                        help='finetune the model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # Model parameters
    parser.add_argument('--model', default='edgenext_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=256, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")
    parser.add_argument('--adl_threshold', type=float, default=0.5)
    parser.add_argument('--adl_keep_prob', type=float, default=0.25)


    args = parser.parse_args()
    global best_acc
    device = torch.device(args.device)

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
            # return torch.Size([15, 21])

        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        elif args.arch == 'pvt_tiny':
            import models.pvt as models
            '''model = models.pvt_tiny(pretrained=True,
                                    num_classes=21,
                                    drop_rate=0.0,
                                    drop_path_rate=0.1)
            '''
            model=models.pvt_tiny()
            path='./pvt_tiny.pth'
            model_dict = model.state_dict()  # 加载模型参数
            pretrain_dict = torch.load(path)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)

        elif args.arch == 'pvt_small':
            import models.pvt as models
            model=models.pvt_small()
            path='./pvt_small.pth'
            model_dict = torch.load(path)  # 加载模型参数
            model.load_state_dict(model_dict)
            for par in model.parameters():  # 冻结部分层的参数
                par.requires_grad = False
            model.head.weight.requires_grad = True

        elif args.arch == 'pvt_large':
            import models.pvt as models
            model=models.pvt_large()
            path='./pvt_large.pth'
            model_dict = torch.load(path)  # 加载模型参数
            model.load_state_dict(model_dict)
            for par in model.parameters():  # 冻结部分层的参数
                par.requires_grad = False
            model.head.weight.requires_grad = True

        elif args.arch == 'pvt_v2_b0':
            import models.pvt_v2 as models
            model = models.pvt_v2_b0()
            path = './pvt_v2_b0.pth'
            model_dict = model.state_dict()  # 加载模型参数
            pretrain_dict = torch.load(path)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict({k.replace('module.',''):v for k,v in model_dict['model'].items()})
            for par in model.parameters():  # 冻结部分层的参数
                par.requires_grad = False
            model.head.weight.requires_grad = True

        elif args.arch == 'pvt_v2_b3' :
            import models.pvt_v2 as models
            model = models.pvt_v2_b3()
            path = './pvt_v2_b3.pth'
            model_dict = model.state_dict()  # 加载模型参数
            pretrain_dict = torch.load(path)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            for k,v in pretrain_dict.items():
                print(k)
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict({k.replace('module.',''):v for k,v in model_dict['model'].items()})
            # for par in model.parameters():  # 冻结部分层的参数
            #     par.requires_grad = False
            # model.head.weight.requires_grad = True

        elif args.arch == 'pvt_v2_b5' :
            import models.pvt_v2 as models
            model = models.pvt_v2_b5()
            path = './pvt_v2_b5.pth'
            model_dict = model.state_dict()  # 加载模型参数
            pretrain_dict = torch.load(path)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict({k.replace('module.',''):v for k,v in model_dict['model'].items()})
            # for par in model.parameters():  # 冻结部分层的参数
            #     par.requires_grad = False
            # model.head.weight.requires_grad = True

        elif args.arch == 'localvit_pvt' :
            import models.localvit_pvt as models
            model = models.localvit_pvt_tiny()
            path = './localvit_pvt.pth'
            model_dict = model.state_dict()  # 加载模型参数
            pretrain_dict = torch.load(path)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict({k.replace('module.',''):v for k,v in model_dict['model'].items()})
            # for par in model.parameters():  # 冻结部分层的参数
            #     par.requires_grad = False
            # model.head.weight.requires_grad = True

        elif args.arch == 'vit_base_patch16_224':
            import models.vit as models

            model = models.vit_base_patch16_224(img_size=256,
                                                num_classes=args.num_classes)
            path = './mae_pretrain_vit_base.pth'
            model_dict = model.state_dict()  # 加载模型参数
            pretrain_dict = torch.load(path)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict({k.replace('module.',''):v for k,v in model_dict['model'].items()})
            for par in model.parameters():  # 冻结部分层的参数
                par.requires_grad = False
            model.head.weight.requires_grad = True

        elif args.arch == 'edgenext_base' :
            import models.edge.model as models
            model = models.edgenext_base(
                classifier_dropout=args.classifier_dropout,
                pretrained=False,
                num_classes=30,
                drop_path_rate=args.drop_path,
                layer_scale_init_value=args.layer_scale_init_value,
                head_init_scale=1.0,
                input_res=args.input_size,
            )
            path = 'edgenext_base_usi.pth'
            # model_dict = model.state_dict()
            # pretrain_dict = torch.load(path)
            # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            # # pretrain_dict.pop('head.weight')
            # # pretrain_dict.pop('head.bias')
            # model_dict.update(pretrain_dict)
            # model.load_state_dict(model_dict)
            # model.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])
            # model_dict = model.state_dict()  # 加载模型参数
            # model.load_state_dict(model_dict)
            model_dict = model.state_dict()  # 加载模型参数
            pretrain_dict = torch.load(path, map_location=device)['state_dict']
            args.crop_pct = 0.95
            pretrain_dict.pop('head.weight')
            pretrain_dict.pop('head.bias')
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if (k in model_dict and 'fc' not in k)}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            # if args.finetune:
            #     checkpoint = torch.load(args.finetune, map_location="cpu")
            #     state_dict = checkpoint[pretrain_dict]
            #     util.load_state_dict(model, state_dict)
            # model.to(device)

        elif args.arch == 'Semiformer':
            import models.semi_transformer as models
            model = models.Semiformer(num_classes=args.num_classes)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model



    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, ")
        # f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'ucm':
        args.num_classes = 21
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 1
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 28
            args.model_width = 8

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset]()

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        # 这里太大了，batch_size=64,mu=7,导致这里是448，然而只有100张图片，
        # drop_last = true,就会把不足448丢掉，就造成了迭代错误
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    # optimizer = adabound.AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)


    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                              print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:  ' + flops)
    print('Params: ' + params)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)


    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    # if args.resume:
    #     logger.info("==> Resuming from checkpoint..")
    #     assert os.path.isfile(
    #         args.resume), "Error: no checkpoint directory found!"
    #     args.out = os.path.dirname(args.resume)
    #     checkpoint = torch.load(args.resume)
    #     best_acc = checkpoint['best_acc']
    #     args.start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     if args.use_ema:
    #         ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])

    if args.resume:
        try:
            logger.info("==> Resuming from checkpoint..")
            assert os.path.isfile(
                args.resume), "Error: no checkpoint directory found!"
            args.out = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            best_acc = checkpoint['best_acc']
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if args.use_ema:
                ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except :
            logger.info("==> Resuming from checkpoint..")
            checkpoint = torch.load(args.resume, map_location='cpu')


    # if args.amp:
    #     from apex import amp
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader, model, optimizer, ema_model, scheduler, p_cutoff=0.0):
    # if args.amp:
    #     from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    # -------warm up
    # selected_label = torch.ones((len(unlabeled_trainloader.dataset),), dtype=torch.long, ) * -1
    # selected_label = selected_label.cuda(args.gpu)
    # classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)
    #
    # pseudo_counter = Counter(selected_label.tolist())
    # if max(pseudo_counter.values()) < len(unlabeled_trainloader.dataset):  # not all(5w) -1
    #     if args.thresh_warmup:
    #         for i in range(args.num_classes):
    #             classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
    #     else:
    #         wo_negative_one = deepcopy(pseudo_counter)
    #         if -1 in wo_negative_one.keys():
    #             wo_negative_one.pop(-1)
    #         for i in range(args.num_classes):
    #             classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

    # --------------------------------warm up

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]


            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)


            from utils.contrastive_loss import contrastive_loss,FocalLoss
            # Lcon = contrastive_loss(logits[:, 0], targets_x.view(-1))
            del logits


            Lcro = F.cross_entropy(logits_x, targets_x, reduction='mean')

            focalLoss = FocalLoss()
            Lfoc = focalLoss(logits_x, targets_x)
            # Lx = 0.6 * Lcro + 0.2 * Lcon + 0.2 * Lfoc
            # Lx = Lcro
            Lx = Lcro +  Lfoc

            '''二进制交叉熵损失'''
            # one_hot = torch.nn.functional.one_hot(targets_x,logits_x[0].size()[0])
            #
            # Lx = F.binary_cross_entropy_with_logits(logits_x, one_hot.float() , reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            #-------------warm up
            # mask = max_probs.ge(p_cutoff * (classwise_acc[targets_u] / (2. - classwise_acc[targets_u]))).float()  # convex
            #-------------end

            mask = max_probs.ge(args.threshold).float()

            # mask = max_probs.le(args.threshold).float()
            Lufoc = (focalLoss(logits_u_s, targets_u) * mask).mean()
            Lu = (F.cross_entropy(logits_u_s, targets_u,reduction='none') * mask).mean() + Lufoc
            # one_hot1= torch.nn.functional.one_hot(targets_u, logits_u_s[0].size()[0])
            # mask = torch.nn.functional.one_hot(mask.to(torch.int64), logits_u_s[0].size()[0])
            # Lu = ( F.binary_cross_entropy_with_logits(logits_u_s, one_hot1.float(), reduction='none') * mask.float()).mean()
            # Lu = F.cross_entropy(logits_u_s, targets_u, reduction='none').mean()

            loss = Lx + args.lambda_u * Lu

            # nnn = Lx.item()
            # mmm = Lu.item()

            # if args.amp:
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    # scaled_loss.backward()
            # else:
                # loss.backward()
            loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.float().mean().item())
            if not args.no_progress:
                # print('loss_x:', losses_x.avg)
                # print('loss_u:', losses_u.avg)
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))
        x.append(epoch + 1)
        y.append(best_acc)
    # fig = plt.figure(figsize=(7, 5))  # figsize调节创建窗口的大小
    # pl.legend()
    # pl.xlabel(u'iters')
    # pl.ylabel(u'acc')
    # plt.title('Compare the data relationship')
    # p1 = pl.plot(np.array(x), np.array(y), 'r', label=u'acc')
    # pl.show()
    # plt.savefig('./data/training.png')

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg



if __name__ == '__main__':
    main()
