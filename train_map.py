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
from network.vgg_map import vgg16
from tensorboardX import SummaryWriter
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import cv2
torch.backends.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='manual seed')
parser.add_argument('--it_start', type=int, default=1, help='number of itr to start with')
parser.add_argument('--it_end', type=int, default=40000, help='number of itr to end with')
parser.add_argument('--signature', default='Test')
parser.add_argument('--model_dir', help='pretrained model')
parser.add_argument('--data_dir', help='directory for data')
parser.add_argument('--save_dir', default='./runs', help='directory for result')
parser.add_argument('--network', default='xcp', help='directory for result')
opt = parser.parse_args()
print(opt)

sig = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + opt.signature
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
os.makedirs('{}/modules/{}'.format(opt.save_dir, sig), exist_ok=True)


class DATA(object):
    def __init__(self, data_root, seed=opt.seed):
        np.random.seed(seed)
        self.data_root = data_root
        self.len = 0
        transform_xcp = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((112, 112)),
                                             transforms.Resize((299, 299)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5] * 3, [0.5] * 3)])

        transform_vgg = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        transform_mask_xcp = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.Resize((19, 19)),
                                                      transforms.Grayscale(num_output_channels=1),
                                                      transforms.ToTensor()])

        transform_mask_vgg = transforms.Compose([transforms.ToPILImage(),
                                                      transforms.Resize((28, 28)),
                                                      transforms.Grayscale(num_output_channels=1),
                                                      transforms.ToTensor()])

        if opt.network == 'xcp':
            self.transform = transform_xcp
            self.transform_mask = transform_mask_xcp
            self.mask_real = torch.zeros((1, 19, 19))
            self.mask_dne = torch.ones((1, 19, 19)) * 100
        if opt.network == 'vgg':
            self.transform = transform_vgg
            self.transform_mask = transform_mask_vgg
            self.mask_real = torch.zeros((1, 28, 28))
            self.mask_dne = torch.ones((1, 28, 28)) * 100

        self.classes = {'Real': 0, 'Fake': 1}
        self.img_paths = {'Real': [], 'Fake': []}

        for f in self.classes.keys():
            dir_list = os.listdir(self.data_root)
            for dir in dir_list:
                file_list = os.listdir(os.path.join(self.data_root, dir, f))
                for i in range(len(file_list)):
                    self.img_paths[f].append(os.path.join(self.data_root, dir, f, file_list[i]))

        for f in self.img_paths.values():
            random.Random(seed).shuffle(f)
            self.len += len(f)

    # Xception
    def __getitem__(self, index):
        img_name_real = random.choice(self.img_paths['Real'])
        img_path_real = img_name_real
        img_real = self.transform(imread(img_path_real))
        mask_path_real = img_path_real.split('/Real/')[0] + '/Mask/' + img_path_real.split('/Real/')[1]
        # mask_real = self.transform_mask(imread(mask_path_real))
        mask_real = self.mask_real

        img_name_fake = random.choice(self.img_paths['Fake'])
        img_path_fake = img_name_fake
        img_fake = self.transform(imread(img_path_fake))
        mask_path_fake = img_path_fake.split('/Fake/')[0] + '/Mask/' + img_path_fake.split('/Fake/')[1]
        try:
            mask_fake = self.transform_mask(imread(mask_path_fake))
        except:
            mask_fake = self.mask_dne
        # print(torch.mean(mask_fake))
        if torch.mean(mask_fake) > 0.005:
            fake_label = 1
        else:
            fake_label = 0
        return img_real, mask_real, 0, img_fake, mask_fake, fake_label


    def __len__(self):
        return self.len


def get_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(), sequence[1].cuda(), sequence[2].cuda(), sequence[3].cuda(), sequence[4].cuda(), sequence[5].cuda()
            yield batch


print("Initializing Data Loader")
train_data = DATA(data_root=(opt.data_dir + 'train/'))
# train_data = DATA(data_root=(opt.data_dir + 'dfdc_train_part_00/'))
train_loader = DataLoader(train_data, num_workers=8, batch_size=opt.batch_size//2, shuffle=True, drop_last=True, pin_memory=True)
training_batch_generator = get_batch(train_loader)

test_data = DATA(data_root=(opt.data_dir + 'validation/'))
# test_data = DATA(data_root=(opt.data_dir + 'dfdc_train_part_00/'))
test_loader = DataLoader(test_data, num_workers=8, batch_size=opt.batch_size//2, shuffle=True, drop_last=True, pin_memory=True)
testing_batch_generator = get_batch(test_loader)


def load_template(index):
    transform_mask_vgg = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((28, 28)),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor()])

    transform_mask_xcp = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((19, 19)),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor()])
    t_list = []
    if index == 0:
        print("Loading Template 0")
        for i in range(10):
            img = imread('./MCT/template{:d}.png'.format(i))

            if opt.network == 'xcp':
                t_list.append(transform_mask_xcp(img).squeeze(0))
            if opt.network == 'vgg':
                t_list.append(transform_mask_vgg(img).squeeze(0))

    elif index == 1:
        print("Loading Template 1")
        for i in range(10):
            img = imread('./ACT/template{:d}.png'.format(i))

            if opt.network == 'xcp':
                t_list.append(transform_mask_xcp(img).squeeze(0))
            if opt.network == 'vgg':
                t_list.append(transform_mask_vgg(img).squeeze(0))

    elif index == 2:
        print("Loading Template 2")
        for i in range(10):
            t_list.append(torch.zeros((19, 19)))

        for i in range(19):
            for j in range(19):
                if 0 <= i < 7 and 0 <= j < 7:
                    t_list[1][i][j] = 1
                if 0 <= i < 7 and 6 <= j < 13:
                    t_list[2][i][j] = 1
                if 0 <= i < 7 and 12 <= j < 19:
                    t_list[3][i][j] = 1
                if 6 <= i < 13 and 0 <= j < 7:
                    t_list[4][i][j] = 1
                if 6 <= i < 13 and 6 <= j < 13:
                    t_list[5][i][j] = 1
                if 6 <= i < 13 and 12 <= j < 19:
                    t_list[6][i][j] = 1
                if 12 <= i < 19 and 0 <= j < 7:
                    t_list[7][i][j] = 1
                if 12 <= i < 19 and 6 <= j < 13:
                    t_list[8][i][j] = 1
                if 12 <= i < 19 and 12 <= j < 19:
                    t_list[9][i][j] = 1

    t = torch.stack(t_list).cuda()
    return t


templates = load_template(1)



print("Initializing Networks")

if opt.network == 'xcp':
    model = xception(templates, len(train_data.classes), True)
if opt.network == 'vgg':
    model = vgg16(templates, len(train_data.classes), True)

if opt.it_start != 1:
    print("Loading Module")
    checkpoint = torch.load(opt.model_dir)
    model.load_state_dict(checkpoint['module'])
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
model.cuda()
cse_loss = nn.CrossEntropyLoss().cuda()
l1_loss = nn.L1Loss().cuda()
if opt.network == 'xcp':
    mp = nn.MaxPool2d(19).cuda()
if opt.network == 'vgg':
    mp = nn.MaxPool2d(28).cuda()

def display(mask_gt, mask_pred):
    mask_gt_cpu = mask_gt.cpu()
    mask_gt_concat = np.concatenate((mask_gt_cpu[0], mask_gt_cpu[1], mask_gt_cpu[2], mask_gt_cpu[3],
                                     mask_gt_cpu[4], mask_gt_cpu[5], mask_gt_cpu[6], mask_gt_cpu[7],
                                     mask_gt_cpu[8], mask_gt_cpu[9], mask_gt_cpu[10], mask_gt_cpu[11],
                                     mask_gt_cpu[12], mask_gt_cpu[13], mask_gt_cpu[14], mask_gt_cpu[15]), axis=2)

    mask_bin_cpu = torch.where(mask_gt_cpu < 0.1, torch.zeros(19, 19), torch.ones(19, 19))
    mask_bin_concat = np.concatenate((mask_bin_cpu[0], mask_bin_cpu[1], mask_bin_cpu[2], mask_bin_cpu[3],
                                      mask_bin_cpu[4], mask_bin_cpu[5], mask_bin_cpu[6], mask_bin_cpu[7],
                                      mask_bin_cpu[8], mask_bin_cpu[9], mask_bin_cpu[10], mask_bin_cpu[11],
                                      mask_bin_cpu[12], mask_bin_cpu[13], mask_bin_cpu[14], mask_bin_cpu[15]), axis=2)

    vec_gt = torch.from_numpy(
        np.dot(mask_bin_cpu.reshape(16, 361).numpy(), np.linalg.pinv(templates.cpu().reshape(10, 361).numpy()))).cuda()
    mask_calc_cpu = torch.mm(vec_gt, templates.reshape(10, 361)).reshape((16, 1, 19, 19)).cpu()
    mask_calc_concat = np.concatenate((mask_calc_cpu[0], mask_calc_cpu[1], mask_calc_cpu[2], mask_calc_cpu[3],
                                       mask_calc_cpu[4], mask_calc_cpu[5], mask_calc_cpu[6], mask_calc_cpu[7],
                                       mask_calc_cpu[8], mask_calc_cpu[9], mask_calc_cpu[10], mask_calc_cpu[11],
                                       mask_calc_cpu[12], mask_calc_cpu[13], mask_calc_cpu[14], mask_calc_cpu[15]),
                                      axis=2)

    mask_pred_cpu = mask_pred.cpu().detach()
    mask_pred_concat = np.concatenate((mask_pred_cpu[0], mask_pred_cpu[1], mask_pred_cpu[2], mask_pred_cpu[3],
                                       mask_pred_cpu[4], mask_pred_cpu[5], mask_pred_cpu[6], mask_pred_cpu[7],
                                       mask_pred_cpu[8], mask_pred_cpu[9], mask_pred_cpu[10], mask_pred_cpu[11],
                                       mask_pred_cpu[12], mask_pred_cpu[13], mask_pred_cpu[14], mask_pred_cpu[15]),
                                      axis=2)

    mask_out = np.concatenate((mask_gt_concat, mask_bin_concat, mask_calc_concat, mask_pred_concat), axis=1)
    mask_out = np.repeat(mask_out.reshape(76, 304, 1), 3, axis=2)
    mask_out = cv2.resize(mask_out, (1216, 304))
    cv2.imshow('mask_out', mask_out)
    # cv2.imwrite("./img_out/00.png", mask_out*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if opt.network == 'xcp':
    mask_dne = torch.ones((1, 19, 19)) * 100
if opt.network == 'vgg':
    mask_dne = torch.ones((1, 28, 28)) * 100

def sup_loss(mask_gt, mask):
    if opt.network == 'xcp':
        mask_exist_index = mask_gt.reshape(16, 361).mean(dim=1)
        zeros = torch.zeros(19, 19)
        ones = torch.ones(19, 19)
    if opt.network == 'vgg':
        mask_exist_index = mask_gt.reshape(16, 784).mean(dim=1)
        zeros = torch.zeros(28, 28)
        ones = torch.ones(28, 28)
    mask_exist_index = (mask_exist_index!=100).nonzero().squeeze()
    mask_exist = torch.index_select(mask, 0, mask_exist_index)
    mask_gt_exist = torch.index_select(mask_gt, 0, mask_exist_index)
    mask_gt_exist_bin = torch.where(mask_gt_exist.cpu() < 0.1, zeros, ones).cuda()
    loss = l1_loss(mask_exist, mask_gt_exist_bin)
    return loss

def train(batch, mask_gt, label):
    model.train()
    x, mask, vec = model(batch)
    loss0 = cse_loss(x, label)
    # map_maxpool = torch.squeeze(mp(mask))
    # print(map_maxpool)
    # loss1 = l1_loss(map_maxpool, label.float())
    # vec_max = torch.max(vec, 1).values
    # vec_mean = torch.mean(vec, 1)
    # loss1 = l1_loss(vec_max, label.float()) + l1_loss(vec_mean, label.float()*0.75)
    loss2 = sup_loss(mask_gt, mask)
    # loss2 = l1_loss(vec, vec_gt)
    loss = loss0 + loss2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return [loss0.item(), 0], 0


def test(batch, mask_gt, label):
    model.eval()
    with torch.no_grad():
        x, mask, vec = model(batch)
        loss0 = cse_loss(x, label)
        # map_maxpool = torch.squeeze(mp(mask))
        # loss1 = l1_loss(map_maxpool, label.float())
        # vec_max = torch.max(vec, 1).values
        # vec_mean = torch.mean(vec, 1)
        # loss1 = l1_loss(vec_max, label.float()) + l1_loss(vec_mean, label.float()*0.75)
        loss2 = sup_loss(mask_gt, mask)
        # loss2 = l1_loss(vec, vec_gt)
        prediction = torch.max(x, dim=1)[1]
        # print(prediction)
        # display(mask_gt, mask)
        # print(mask_exist.reshape(mask_exist.shape[0], 361).mean(dim=1))
        # print(mask_gt_exist.reshape(mask_gt_exist.shape[0], 361).mean(dim=1))
        accu = (prediction.eq(label.long())).sum()
    return [loss0.item(), loss2.item(), accu.item() / len(batch)], 0


def write_tfboard(vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)


writer = SummaryWriter('%s/logs/%s' % (opt.save_dir, sig))
itr = opt.it_start
print("Start Training at iteration {:d}".format(itr))
while itr != opt.it_end + 1:
    batch_real_train, mask_real_train, label_real_train, batch_fake_train, mask_fake_train, label_fake_train = next(
        training_batch_generator)
    batch_train = torch.cat((batch_real_train, batch_fake_train), 0)
    mask_train = torch.cat((mask_real_train, mask_fake_train), 0)
    label_train = torch.cat((label_real_train, label_fake_train), 0)
    loss, mask = train(batch_train, mask_train, label_train)
    write_tfboard(loss, itr, name='TRAIN')

    if itr % 100 == 0:
        batch_real_test, mask_real_test, label_real_test, batch_fake_test, mask_fake_test, label_fake_test = next(
            testing_batch_generator)
        batch_test = torch.cat((batch_real_test, batch_fake_test), 0)
        mask_test = torch.cat((mask_real_test, mask_fake_test), 0)
        label_test = torch.cat((label_real_test, label_fake_test), 0)
        lossacc, mask = test(batch_test, mask_test, label_test)
        x1 = vutils.make_grid(batch_test, normalize=True, scale_each=True)
        # x2 = vutils.make_grid(mask, normalize=True, scale_each=True)
        writer.add_image('Image_orig', x1, itr)
        # writer.add_image('Image_map', x2, itr)
        writer.add_text('Text', 'Image Label: ' + str(label_test.tolist()), itr)
        write_tfboard(lossacc, itr, name='TEST')
        print("Eval: {:d}".format(itr))

    if itr % 10000 == 0:
        torch.save({'module': model.state_dict()}, '%s/modules/%s/%d.pickle' % (opt.save_dir, sig, itr))
        print("Save Model: {:d}".format(itr))

    itr += 1
