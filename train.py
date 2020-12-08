import numpy as np
import os
import random
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import Dataset
from templates import get_templates

MODEL_DIR = './models/'
BACKBONE = 'xcp'
MAPTYPE = 'tmp'
BATCH_SIZE = 15
MAX_EPOCHS = 100
STEPS_PER_EPOCH = 1000
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001

CONFIGS = {
  'xcp': {
          'img_size': (299, 299),
          'map_size': (19, 19),
          'norms': [[0.5] * 3, [0.5] * 3]
         },
  'vgg': {
          'img_size': (299, 299),
          'map_size': (19, 19),
          'norms': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
         }
}
CONFIG = CONFIGS[BACKBONE]

if BACKBONE == 'xcp':
  from xception import Model
elif BACKBONE == 'vgg':
  from vgg import Model

torch.backends.deterministic = True
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_TRAIN = Dataset('train', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)

DATA_EVAL = Dataset('eval', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)

TEMPLATES = None
if MAPTYPE in ['tmp', 'pca_tmp']:
  TEMPLATES = get_templates()

MODEL_NAME = '{0}_{1}'.format(BACKBONE, MAPTYPE)
MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'

MODEL = Model(MAPTYPE, TEMPLATES, 2, False)

OPTIM = optim.Adam(MODEL.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
MODEL.model.cuda()
LOSS_CSE = nn.CrossEntropyLoss().cuda()
LOSS_L1 = nn.L1Loss().cuda()
MAXPOOL = nn.MaxPool2d(19).cuda()

def calculate_losses(batch):
  img = batch['img']
  msk = batch['msk']
  lab = batch['lab']
  x, mask, vec = MODEL.model(img)
  loss_l1 = LOSS_L1(mask, msk)
  loss_cse = LOSS_CSE(x, lab)
  loss = loss_l1 + loss_cse
  pred = torch.max(x, dim=1)[1]
  acc = (pred == lab).float().mean()
  return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'acc': acc }

def process_batch(batch, mode):
  if mode == 'train':
    MODEL.model.train()
    losses = calculate_losses(batch)
    OPTIM.zero_grad()
    losses['loss'].backward()
    OPTIM.step()
  elif mode == 'eval':
    MODEL.model.eval()
    with torch.no_grad():
      losses = calculate_losses(batch)
  return losses

SUMMARY_WRITER = SummaryWriter(MODEL_DIR + 'logs/')
def write_tfboard(item, itr, name):
  SUMMARY_WRITER.add_scalar('{0}'.format(name), item, itr)

def run_step(e, s):
  batch = DATA_TRAIN.get_batch()
  losses = process_batch(batch, 'train')

  if s % 10 == 0:
    print('\r{0} - '.format(s) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
  if s % 100 == 0:
    print('\n', end='')
    [write_tfboard(losses[_], e * STEPS_PER_EPOCH + s, _) for _ in losses]

def run_epoch(e):
  print('Epoch: {0}'.format(e))
  for s in range(STEPS_PER_EPOCH):
    run_step(e, s)
  MODEL.save(e+1, OPTIM, MODEL_DIR)

LAST_EPOCH = 0
for e in range(LAST_EPOCH, MAX_EPOCHS):
  run_epoch(e)

