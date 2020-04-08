import argparse
import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from imageio import imread
from network.xception_map import xception
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
torch.backends.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='manual seed')
parser.add_argument('--it_start', type=int, default=1, help='number of itr to start with')
parser.add_argument('--it_end', type=int, default=40000, help='number of itr to end with')
parser.add_argument('--signature', default=str(datetime.datetime.now()))
parser.add_argument('--data_dir', help='directory for data')
parser.add_argument('--save_dir', default='./runs', help='directory for result')
opt = parser.parse_args()
print(opt)

sig = str(datetime.datetime.now()) + opt.signature
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
os.makedirs('%s/modules/%s' % (opt.save_dir, sig), exist_ok=True)


class DATA(object):
    def __init__(self, data_root, seed=opt.seed):
        np.random.seed(seed)
        self.data_root = data_root

        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((299, 299)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5] * 3, [0.5] * 3)])

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((19, 19)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        self.classes = {'Real': 0, 'Fake': 1}
        self.img_paths = []

        file_names = os.listdir('{:s}images/'.format(self.data_root))
        for file_name in file_names:
            self.img_paths.append(file_name)


    def load_img(self, path):
        img = imread(path)
        img = self.transform(img)
        return img

    def load_mask(self, path):
        img = imread(path)
        img = self.transform_mask(img)

        return img


    def __getitem__(self, index):
        image_path_real = '{:s}images/{:0>5d}-0.jpg'.format(self.data_root, index)
        img_real0 = self.load_img(image_path_real)
        image_path_fake1 = '{:s}images/{:0>5d}-1.jpg'.format(self.data_root, index)
        img_fake1 = self.load_img(image_path_fake1)
        image_path_fake2 = '{:s}images/{:0>5d}-2.jpg'.format(self.data_root, index)
        img_fake2 = self.load_img(image_path_fake2)
        image_path_fake3 = '{:s}images/{:0>5d}-3.jpg'.format(self.data_root, index)
        img_fake3 = self.load_img(image_path_fake3)
        image_path_mask0 = '{:s}Mask/{:0>5d}-0.jpg'.format(self.data_root, index)
        img_mask0 = self.load_mask(image_path_mask0)
        image_path_mask1 = '{:s}Mask/{:0>5d}-1.jpg'.format(self.data_root, index)
        img_mask1 = self.load_mask(image_path_mask1)
        image_path_mask2 = '{:s}Mask/{:0>5d}-2.jpg'.format(self.data_root, index)
        img_mask2 = self.load_mask(image_path_mask2)
        image_path_mask3 = '{:s}Mask/{:0>5d}-3.jpg'.format(self.data_root, index)
        img_mask3 = self.load_mask(image_path_mask3)

        return img_real0, img_fake1, img_fake2, img_fake3, img_mask0, img_mask1, img_mask2, img_mask3, 0, 1, 1, 1

    def __len__(self):
        return len(self.img_paths)//4


def get_training_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(), sequence[1].cuda(), sequence[2].cuda(), sequence[3].cuda(), sequence[4].cuda(), sequence[5].cuda(), \
                    sequence[6].cuda(), sequence[7].cuda(), sequence[8].cuda(), sequence[9].cuda(), sequence[10].cuda(), sequence[11].cuda()
            yield batch


print("Initializing Data Loader")
train_data = DATA(data_root=(opt.data_dir + 'train/'))
train_loader = DataLoader(train_data, num_workers=8, batch_size=opt.batch_size//4, shuffle=True, drop_last=True, pin_memory=True)
training_batch_generator = get_training_batch(train_loader)

test_data = DATA(data_root=(opt.data_dir + 'validation/'))
test_loader = DataLoader(test_data, num_workers=8, batch_size=opt.batch_size//4, shuffle=True, drop_last=True, pin_memory=True)
testing_batch_generator = get_training_batch(test_loader)

print("Loading Templates")
templates_list = []
for i in range(10):
    img = imread('./MCT/template{:d}.png'.format(i))
    templates_list.append(transforms.functional.to_tensor(img)[0:1,0:19,0:19])
templates = torch.stack(templates_list).cuda()
templates = templates.squeeze(1)
templates = templates.repeat(16, 1, 1, 1)

print("Initializing Networks")
model = xception(templates, len(train_data.classes), True)
# checkpoint = torch.load('./90000.pickle')
# model.load_state_dict(checkpoint['module'])
optimizer_xcp = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.001)
model.cuda()
cse_loss = nn.CrossEntropyLoss().cuda()
l1_loss = nn.L1Loss().cuda()
mp = nn.MaxPool2d(19).cuda()


def train_xcp(batch, label, mask_gt):
    model.train()
    x, mask, vec = model(batch)
    loss1 = l1_loss(mask, mask_gt)
    loss2 = cse_loss(x, label)
    loss = loss1 + loss2
    optimizer_xcp.zero_grad()
    loss.backward()
    optimizer_xcp.step()
    return [loss2.item(), loss1.item()], mask


def test(batch, label, mask_gt):
    model.eval()
    with torch.no_grad():
        x, mask, vec = model(batch)
        loss1 = l1_loss(mask, mask_gt)
        loss2 = cse_loss(x, label)
        prediction = torch.max(x, dim=1)[1]
        accu = (prediction.eq(label.long())).sum()
    return [loss2.item(), loss1.item(), accu.item()/len(batch)], mask


def write_tfboard(vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)


writer = SummaryWriter('%s/logs/%s' % (opt.save_dir, sig))
itr = opt.it_start
print("Start Training at iteration {:d}".format(itr))
while itr != opt.it_end+1:
    batch_real0_train, batch_fake1_train, batch_fake2_train, batch_fake3_train, batch_mask0_train, batch_mask1_train, batch_mask2_train, batch_mask3_train, label_real_train, label_fake_train, label_fake_train, label_fake_train = next(training_batch_generator)
    batch_train = torch.cat((batch_real0_train, batch_fake1_train, batch_fake2_train, batch_fake3_train), 0)
    label_train = torch.cat((label_real_train, label_fake_train, label_fake_train, label_fake_train), 0)
    mask_train = torch.cat((batch_mask0_train, batch_mask1_train, batch_mask2_train, batch_mask3_train), 0)
    loss, mask = train_xcp(batch_train, label_train, mask_train)
    write_tfboard(loss, itr, name='TRAIN')

    if itr % 100 == 0:
        batch_real0_test, batch_fake1_test, batch_fake2_test, batch_fake3_test, batch_mask0_test, batch_mask1_test, batch_mask2_test, batch_mask3_test, label_real_test, label_fake_test, label_fake_test, label_fake_test = next(testing_batch_generator)
        batch_test = torch.cat((batch_real0_test, batch_fake1_test, batch_fake2_test, batch_fake3_test), 0)
        label_test = torch.cat((label_real_test, label_fake_test, label_fake_test, label_fake_test), 0)
        mask_test = torch.cat((batch_mask0_test, batch_mask1_test, batch_mask2_test, batch_mask3_test), 0)
        lossacc, mask = test(batch_test, label_test, mask_test)
        x1 = vutils.make_grid(batch_test, normalize=True, scale_each=True)
        x2 = vutils.make_grid(mask, normalize=True, scale_each=True)
        writer.add_image('Image_orig', x1, itr)
        writer.add_image('Image_map', x2, itr)
        writer.add_text('Text', 'Image Label: ' + str(label_test.tolist()), itr)

        write_tfboard(lossacc, itr, name='TEST')
        print("Eval: {:d}".format(itr))
    if itr % 10000 == 0:
        torch.save({'module': model.state_dict()}, '%s/modules/%s/%d.pickle' % (opt.save_dir, sig, itr))
        print("Save Model: {:d}".format(itr))

    itr += 1
