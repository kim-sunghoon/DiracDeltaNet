import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import time

from models import ShuffleNetv2_wrapper
from models import DiracDeltaNet_wrapper

from torch.autograd import Variable
from extensions.utils import progress_bar
from extensions.utils import create_dir
from extensions.model_refinery_wrapper import ModelRefineryWrapper
from extensions.refinery_loss import RefineryLoss

from r_utils.config_helpers import merge_configs


def select_dataset_config():
    from general_config import cfg as general_cfg
    ## for the CIFAR10 dataset use:
    from r_utils.configs.cifar10_config import cfg as dataset_cfg
    ## for the MNIST dataset use:
    #  from r_utils.configs.mnist_config import cfg as dataset_cfg
    ## for the CIFAR100 dataset use:
    #  from r_utils.configs.cifar100_config import cfg as dataset_cfg
    ## for the ImageNet  dataset use:
    #  from r_utils.configs.ImageNet_config import cfg as dataset_cfg

    return merge_configs([general_cfg, dataset_cfg])

## TODO: imagetnet일때문 RefineryLoss를 사용할 수 있다. XNOR쓴 그룹에서 이런걸 내놨었네. 다른 dataset일때 loss를 바꿔야함
num_classes=1000

best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def parse_opts():
    parser = argparse.ArgumentParser(description='PyTorch imagenet Training in quant')
    
    parser.add_argument('--datadir', help='path to dataset')
    parser.add_argument('--inputdir', help='path to input model')
    parser.add_argument('--outputdir', help='path to output model')
    parser.add_argument('--logdir', default='./log/log.csv', help='path to log')
    
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    
    parser.add_argument('--lr', default=0.5, type=float, help='learning rate')
    parser.add_argument('--lr_policy', choices=('step', 'linear'), help='learning rate decay policy', default='linear')
    parser.add_argument('--totalepoch', default=90, type=int, help='how many epoch')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='batch size')
    parser.add_argument('--test_batch_size', '-tb', default=32, type=int, help='test_batch size')
    parser.add_argument('--weight_decay', '--wd', default=4e-5, type=float, help='weight decay (default: 4e-5)')
    parser.add_argument('--crop_scale', default=0.2, type=float, help='random resized crop scale')
    
    parser.add_argument('--expansion', '-e', default=2.0, type=float, help='expansion rate for the middle plate')
    parser.add_argument('--base_channel_size', default=116, type=int, help='base channel size of the shuffle block')
    parser.add_argument('--weight_bit', default=32, type=int, help='conv weight bitwidth')
    parser.add_argument('--act_bit', default=32, type=int, help='activation bitwidth')
    parser.add_argument('--first_weight_bit', default=32, type=int, help='first conv weight bitwidth')
    parser.add_argument('--first_act_bit', default=32, type=int, help='first conv activation bitwidth')
    parser.add_argument('--last_weight_bit', default=32, type=int, help='last conv weight bitwidth')
    parser.add_argument('--last_act_bit', default=32, type=int, help='last conv activation bitwidth')
    parser.add_argument('--fc_bit', default=32, type=int, help='fc weight bitwidth')
    parser.add_argument('--n_gpu', default = "0,1,2,3", type = str, help = 'specify gpu #')
    
    args = parser.parse_args()

    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: 왜 multigpu할때 device_ids를 해줘야하지... torch.cuda.device_count() 내용도 바뀌지가 않는다. 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu

    print("cuda visible devices: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    args.MGPU = True if len(list(map(int, args.n_gpu.split(',')))) > 1 else False
    
    return vars(args)



# Data
def data_set_config_origin(args):
    print('==> Preparing data..')
    # Data loading code
    traindir = os.path.join(args.datadir, 'train')
    valdir = os.path.join(args.datadir, 'val')
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_train=transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(args.crop_scale,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    #imagenet
    trainset = datasets.ImageFolder(traindir, transform_train)
    testset = datasets.ImageFolder(valdir, transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=30)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True, num_workers=30)
    return trainloader, testloader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def adjust_learning_rate(optimizer, iteration, lr):
    if args.lr_policy == 'linear' :
        #linear lr decay
        total_iteration=(int(1281167/args.batch_size)+1)*args.totalepoch
        new_lr=lr-lr*float(iteration)/(float(total_iteration-1.0))

    else:
        #step lr decay for fine tuning
        if epoch<20:
            new_lr=lr
        elif epoch<30:
            new_lr=lr/5.0
        elif epoch<40:
            new_lr=lr/25.0
        else:
            new_lr=lr/125.0

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


# Training
def train(epoch, net, optimizer, criterion, args):
    global iteration

    print('\nEpoch: %d' % epoch)
    net.train()
    criterion.train()
    net.to(args.device)
    train_loss = 0
    correct_1 = 0 # moniter top 1
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        adjust_learning_rate(optimizer, iteration, args.lr)
        if args.use_cuda:
            inputs, targets = inputs.cuda(args.device, non_blocking=True), targets.cuda(args.device,non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        if isinstance(loss, tuple):
            loss_value, outputs = loss
        else:
            loss_value = loss
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        total += targets.size(0)
        correct_1 += prec1
        correct_5 += prec5

        iteration=iteration+1

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct_1)/float(total), correct_1, total))
    return 100.*float(correct_1)/float(total),100.*float(correct_5)/float(total),train_loss

def test(epoch, net, criterion, args):
    global best_acc
    net.eval()
    criterion.eval()
    test_loss = 0
    correct_1 = 0
    correct_5 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if args.use_cuda:
            inputs, targets = inputs.cuda(args.device), targets.cuda(args.device)
        with torch.no_grad():
            outputs = net(inputs)
        loss = criterion(outputs, targets)

        if isinstance(loss, tuple):
            loss_value, outputs = loss
        else:
            loss_value = loss

        test_loss += loss_value.item()
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        total += targets.size(0)
        correct_1 += prec1
        correct_5 += prec5

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct_1)/float(total), correct_1, total))

    # Save checkpoint.
    acc_1 = 100.*float(correct_1)/float(total)
    if acc_1 > best_acc:
        print('Saving..')
        state = {
            'net': net.module.model if args.use_cuda and args.MGPU else net.model,
            'acc_1': acc_1,
            'acc_5': 100.*float(correct_5)/float(total),
            'lr': args.lr,
            'epoch': epoch,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
        }
        torch.save(state, output_path)
        print('* Saved checkpoint to %s' % output_path)
        best_acc = acc_1

    return 100.*float(correct_1)/float(total),100.*float(correct_5)/float(total),test_loss

def load_cnn(args):
    global best_acc
    global start_epoch

    if args.inputdir != None:
        input_path = args.inputdir
        print('Using input path: %s to fine tune' % input_path)
        checkpoint = torch.load(input_path)
        init_net = checkpoint['net']
        init_net=init_net.to('cpu')
    else:
        if args.resume:
            print('Resuming from checkpoint')
        else:
            print('Training from scratch')
    if not args.resume:
        if args.inputdir != None:
            net=DiracDeltaNet_wrapper(expansion=args.expansion, base_channelsize=args.base_channel_size, num_classes=num_classes, 
                weight_bit=args.weight_bit, act_bit=args.act_bit, first_weight_bit=args.first_weight_bit, first_act_bit=args.first_act_bit, 
                last_weight_bit=args.last_weight_bit, last_act_bit=args.last_act_bit, fc_bit=args.fc_bit, extern_init=True, init_model=init_net)
        else:
            net=DiracDeltaNet_wrapper(expansion=args.expansion, base_channelsize=args.base_channel_size, num_classes=num_classes, 
                weight_bit=args.weight_bit, act_bit=args.act_bit, first_weight_bit=args.first_weight_bit, first_act_bit=args.first_act_bit, 
                last_weight_bit=args.last_weight_bit, last_act_bit=args.last_act_bit, fc_bit=args.fc_bit)
    else:
        checkpoint = torch.load(output_path)
        net = checkpoint['net']
        best_acc = checkpoint['acc_1']
        start_epoch = checkpoint['epoch']+1
    
    # reuse sturcture
    label_refinery=torch.load('./resnet50.t7')
    net = ModelRefineryWrapper(net, label_refinery)
    
    #  if torch.cuda.device_count() > 1:
    if args.MGPU:
        print("Let's use", args.n_gpu, "GPUs!")
        net = nn.DataParallel(net, device_ids = list(map(int, args.n_gpu.split(','))))
    net=net.to(args.device)
    return net


if __name__ == "__main__":
    args = parse_opts()
    cfg = select_dataset_config()
    args = merge_configs([cfg, args])

    print(args)

    create_dir(os.path.dirname(args.outputdir))
    output_path = args.outputdir
    print('Using output path: {}'.format(output_path))

    trainloader, testloader = data_set_config_origin(args)
    criterion = RefineryLoss()

    net = load_cnn(args)

    model_trainable_parameters = filter(lambda x: x.requires_grad, net.parameters())
    optimizer = optim.SGD(model_trainable_parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if not args.resume:
        iteration = 0
    else:
        iteration = start_epoch*(int(1281167/args.batch_size)+1)

    create_dir(os.path.dirname(args.logdir))
    if not args.resume:
        f=open(args.logdir,'w')
        f.write("Epoch, tr-top1, tr-top5, tr-loss, tt-top1, tt-top5, tt-loss \n")
        f.close()

    start = time.time()
    for epoch in range(start_epoch, int(args.totalepoch)):
        f=open(args.logdir,'a')

        acc1,acc5,loss=train(epoch, net, optimizer, criterion, args)
        f.write("{}, {}, {}, {},".format(str(epoch), str(acc1), str(acc5), str(loss)))

        acc1,acc5,loss=test(epoch, net, criterion, args)
        f.write("{}, {}, {} \n".format(str(acc1), str(acc5), str(loss)))
        f.close()
    end = (time.time() - start) // 60

    print("train time: {}D {}H {}M".format(end//1440, (end%1440)//60, end%60))
