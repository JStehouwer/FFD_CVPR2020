import numpy as np
import os
import random
from scipy.io import savemat
import shutil
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import Dataset
from templates import get_templates

MODEL_DIR = './models/'
BACKBONE = 'xcp'
MAPTYPE = 'tmp'
BATCH_SIZE = 200
MAX_EPOCHS = 100

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

def get_dataset():
  return Dataset('test', BATCH_SIZE, CONFIG['img_size'], CONFIG['map_size'], CONFIG['norms'], SEED)

DATA_TEST = None

TEMPLATES = None
if MAPTYPE in ['tmp', 'pca_tmp']:
  TEMPLATES = get_templates()

MODEL_NAME = '{0}_{1}'.format(BACKBONE, MAPTYPE)
MODEL_DIR = MODEL_DIR + MODEL_NAME + '/'

MODEL = Model(MAPTYPE, TEMPLATES, 2, False)

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
  res = { 'lab': lab, 'msk': msk, 'score': x, 'pred': pred, 'mask': mask }
  results = {}
  for r in res:
    results[r] = res[r].squeeze().cpu().numpy()
  return { 'loss': loss, 'loss_l1': loss_l1, 'loss_cse': loss_cse, 'acc': acc }, results

def process_batch(batch, mode):
  MODEL.model.eval()
  with torch.no_grad():
    losses, results = calculate_losses(batch)
  return losses, results

def run_step(di, e, s, resultdir):
  batch = DATA_TEST.get_batch(di)
  if batch is None:
    return True
  losses, results = process_batch(batch, 'test')

  savemat('{0}{1}_{2}.mat'.format(resultdir, di, s), results)

  if s % 10 == 0:
    print('\r{0} - '.format(s) + ', '.join(['{0}: {1:.3f}'.format(_, losses[_].cpu().detach().numpy()) for _ in losses]), end='')
  return False

def run_epoch(di, e, resultdir):
  s = 0
  while True:
    s += 1
    is_done = run_step(di, e, s, resultdir)
    if is_done:
      break

LAST_EPOCH = 75
for e in range(LAST_EPOCH, MAX_EPOCHS, 5):
  resultdir = '{0}results/{1}/'.format(MODEL_DIR, e)
  if os.path.exists(resultdir):
    shutil.rmtree(resultdir)
  os.makedirs(resultdir, exist_ok=True)
  MODEL.load(e, MODEL_DIR)
  DATA_TEST = get_dataset()
  for di, d in enumerate(DATA_TEST.datasets):
    run_epoch(di, e, resultdir)
  print()

print('Testing complete')
