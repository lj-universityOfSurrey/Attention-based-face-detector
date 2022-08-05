from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc
from data.config import cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
from models.loss import focal_loss
from balance import BalancedDataParallel

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset',
                    default='./train/label.txt',
                    help='Training dataset directory')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=7, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder',
                    default='./weights/',
                    help='Location to save checkpoint models')


args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = cfg_re50
rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if args.resume_net is not None:
    print('Loading resume network...')
    net = load_model(net, args.resume_net, False)


if num_gpu > 1 and gpu_train:
   net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size, cfg['decay3'] * epoch_size)

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        if args.resume_epoch * epoch_size <= stepvalues[0]:
            step_index = 0
        elif stepvalues[0] < args.resume_epoch * epoch_size <= stepvalues[1]:
            step_index = 1
        elif args.resume_epoch * epoch_size > stepvalues[1]:
            step_index = 2
    else:
        start_iter = 0
        step_index = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,collate_fn=detection_collate))
            if (epoch % 30 == 0 and epoch > 0) or (epoch % 30 == 0 and epoch > cfg['decay2']):
                torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1
        if iteration % 200 == 0:
            torch.cuda.empty_cache()
#        pdb.set_trace()
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr_map, lr_backbone, lr_base = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
       
        # load train data
        images, maps, targets = next(batch_iterator)
#        pdb.set_trace()
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        maps = maps.cuda()

        # forward
        Premap, out = net(images)

        # backprop
#        pdb.set_trace()
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss_1 = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss_2 = focal_loss(Premap, maps)
        loss = loss_1 + loss_2 * 6
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || map: {:.4f} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} '
              '|| LR_Map: {:.8f} LR_bb: {:.8f} LR_Base: {:.8f}|| Batchtime: {:.4f} s || ETA: {} '
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_2.item(), loss_l.item(), loss_c.item(), loss_landm.item(), lr_map, lr_backbone,
                      lr_base, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = 5
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))

    for i, param_group in enumerate(optimizer.param_groups):
        if i == 1:
            param_group['lr'] = 0
            lr_backbone = param_group['lr']
        else:
            param_group['lr'] = lr
    return lr, lr_backbone, lr

if __name__ == '__main__':
    train()
