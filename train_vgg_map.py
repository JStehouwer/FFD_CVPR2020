import argparse
import datetime
import numpy as np
import os
import random
import torch
print(torch.cuda.is_available())
import torch.nn as nn
import torch.optim as optim
import torchvision
from imageio import imread
from network.vgg_map import vgg16
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
torch.backends.deterministic = True
import scipy.io
#################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=5)
parser.add_argument('--it_start', type=int, default=1, help='number of itr to start with')
parser.add_argument('--it_end', type=int, default=-1, help='number of itr to end with')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--train_data_root', default='/home/hao/HotData/v4_mask/train/', help='root directory for data')
parser.add_argument('--test_data_root', default = '/home/hao/HotData/v4_mask/validation/', help='root directory for data')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size_real', default=6, type=int, help='batch size real')
parser.add_argument('--batch_size_fake', default=10, type=int, help='batch size fake')
parser.add_argument('--class_num', default=2, type=int, help='class number')
parser.add_argument('--signature', default=str(datetime.datetime.now()))
parser.add_argument('--savedir', default='./runs')
opt = parser.parse_args()
print(opt)

sig = str(datetime.datetime.now()) + opt.signature
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
os.makedirs('%s/modules/%s' % (opt.savedir, sig), exist_ok=True)
#################################################################################################################



class RealData(object):
    def __init__(self, data_root, image_width=224, image_height=224, seed=opt.seed):

        np.random.seed(seed)
        self.data_root = data_root
        self.image_width = image_width
        self.image_height = image_height

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_width, image_height)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        self.classes = {'Real': 1, 'Fake': 0}
        self.img_paths = {'Real': [], 'Fake': []}


        for f in self.classes.items():
            file_names = os.listdir(os.path.join(self.data_root, f[0]))
            for file_name in file_names:
                self.img_paths[f[0]].append((os.path.join(f[0], file_name), f[1]))

        for f in self.img_paths.values():
            random.Random(seed).shuffle(f)

    def load_img(self, path):
        img = imread(path)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        random_folder_1 = 'Real'
        label1 = self.classes[random_folder_1]

        img_name = random.choice(self.img_paths[random_folder_1])[0]
        image_path1 = os.path.join(self.data_root, img_name)
        img1 = self.load_img(image_path1)
        mask1 = torch.zeros(1, 28, 28)
        return img1, img_name, label1, mask1

    def __len__(self):
        return sum([len(i) for i in self.img_paths.values()])

class FakeData(object):
    def __init__(self, data_root, image_width=224, image_height=224, seed=opt.seed):

        np.random.seed(seed)
        self.data_root = data_root
        self.image_width = image_width
        self.image_height = image_height

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_width, image_height)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])


        self.classes = {'Real': 1, 'Fake': 0}
        self.img_paths = {'Real': [], 'Fake': []}


        for f in self.classes.items():
            file_names = os.listdir(os.path.join(self.data_root, f[0]))
            for file_name in file_names:
                self.img_paths[f[0]].append((os.path.join(f[0], file_name), f[1]))

        for f in self.img_paths.values():
            random.Random(seed).shuffle(f)

    def load_img(self, path):
        img = imread(path)
        img = self.transform(img)
        return img

    def load_mask(self, path):
        img = imread(path)
        img = self.mask_transform(img)
        img = torch.where(img < 0.1, torch.zeros(28, 28), torch.ones(28, 28))
        return img

    def __getitem__(self, index):
        random_folder_1 = 'Fake'
        label1 = self.classes[random_folder_1]

        img_name = random.choice(self.img_paths[random_folder_1])[0]
        mask_name = 'Fake_mask' + img_name[4:] 

        image_path1 = os.path.join(self.data_root, img_name)
        img1 = self.load_img(image_path1)
        mask_path1 = os.path.join(self.data_root, mask_name)
        if os.path.exists(mask_path1):
            mask1 = self.load_mask(mask_path1)
            mask1 = mask1[0,:,:].unsqueeze(0)
        else:
            mask1 = torch.zeros(1, 28, 28)

        return img1, img_name, label1, mask1

    def __len__(self):
        return sum([len(i) for i in self.img_paths.values()])


def get_training_real_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(), sequence[1], sequence[2].cuda(), sequence[3].cuda()
            yield batch

def get_training_fake_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(), sequence[1], sequence[2].cuda(), sequence[3].cuda()
            yield batch


print("Initializing Data Loader")
train_real_data = RealData(data_root=opt.train_data_root)
train_fake_data = FakeData(data_root=opt.train_data_root)
train_real_loader = DataLoader(train_real_data, num_workers=4, batch_size=opt.batch_size_real, shuffle=True, drop_last=True, pin_memory=True)
train_fake_loader = DataLoader(train_fake_data, num_workers=4, batch_size=opt.batch_size_fake, shuffle=True, drop_last=True, pin_memory=True)
training_batch_real_generator = get_training_real_batch(train_real_loader)
training_batch_fake_generator = get_training_fake_batch(train_fake_loader)

test_real_data = RealData(data_root=opt.test_data_root)
test_fake_data = FakeData(data_root=opt.test_data_root)
test_real_loader = DataLoader(test_real_data, num_workers=4, batch_size=opt.batch_size_real, shuffle=True, drop_last=True, pin_memory=True)
test_fake_loader = DataLoader(test_fake_data, num_workers=4, batch_size=opt.batch_size_fake, shuffle=True, drop_last=True, pin_memory=True)
testing_batch_real_generator = get_training_real_batch(test_real_loader)
testing_batch_fake_generator = get_training_fake_batch(test_fake_loader)
#################################################################################################################


# # # # #
print("Initializing Networks")
model_xcp = vgg16(opt.class_num)
optimizer_xcp = optim.Adam(model_xcp.parameters(), lr=opt.lr)
#optimizer_xcp = optim.SGD(model_xcp.parameters(), lr=opt.lr, momentum=0.9)
model_xcp.cuda()


cse_loss = nn.CrossEntropyLoss()
cse_loss.cuda()
#################################################################################################################
def pca_loss(pca_map, mask, label):

    real_index = (label==1).nonzero().squeeze()
    fake_index = (label==0).nonzero().squeeze()
    real_pca_map = torch.index_select(pca_map,0,real_index)
    real_mask = torch.index_select(mask,0, real_index)
    fake_pca_map = torch.index_select(pca_map,0,fake_index)
    fake_mask = torch.index_select(mask,0, fake_index)
    f_sum = torch.sum(fake_mask.view(fake_mask.shape[0], fake_mask.shape[1]*fake_mask.shape[2]*fake_mask.shape[3]),1)
    f_idx1 = (f_sum!=0).nonzero().squeeze()
    #f_idx2 = (f_sum==0).nonzero().squeeze()
    f_mask1 = torch.index_select(fake_mask,0,f_idx1)
    f_pca_map1 = torch.index_select(fake_pca_map,0,f_idx1)
    #f_pca_map2 = torch.index_select(fake_pca_map,0,f_idx2)
    f_loss1 = torch.mean(torch.abs(f_pca_map1-f_mask1))
    #f_loss2 = torch.mean(torch.abs(torch.max(f_pca_map2,dim=1)[0]-0.75))
    r_loss = torch.mean(torch.abs(real_pca_map-real_mask))

    #p_loss = torch.mean(f_loss1+f_loss2+r_loss)
    p_loss = torch.mean(f_loss1+r_loss)
    return p_loss, real_pca_map, fake_pca_map


def train_xcp_orig(batch, label, mask):
    model_xcp.train()
    map_, y = model_xcp(batch)
    p_loss, real_pca_map, fake_pca_map = pca_loss(map_, mask, label)
    c_loss = cse_loss(y, label)
    loss = c_loss + p_loss

    optimizer_xcp.zero_grad()
    loss.backward()
    optimizer_xcp.step()
    return[c_loss.item()], [p_loss.item()], real_pca_map, fake_pca_map


def test_orig(batch, label):
    model_xcp.eval()            # Change
    x_, n = model_xcp(batch)        # Change
    loss = cse_loss(n, label)
    prediction = torch.max(n, dim=1)[1]
    accu = (prediction.eq(label.long())).sum()
    return [loss.item(), accu.item()/len(batch)]

#################################################################################################################


def write_tfboard(vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)


writer = SummaryWriter('%s/logs/%s' % (opt.savedir, sig))
itr = opt.it_start


while itr != opt.it_end+1:
    print("Iteration: " + str(itr))
    batch_real, real_name, lb_real, mask_real = next(training_batch_real_generator)
    batch_fake, fake_name, lb_fake, mask_fake = next(training_batch_fake_generator)

    batch = torch.cat((batch_real, batch_fake), 0)
    label = torch.cat((lb_real, lb_fake), 0)
    mask = torch.cat((mask_real, mask_fake), 0)
    img_name = real_name + fake_name

    #idx = torch.randperm(opt.batch_size_real+opt.batch_size_fake)
    #idx_np = idx.numpy()
    #batch = batch[idx, :, :, :]
    #mask = mask[idx, :, :, :]
    #label = label[idx]

    c_loss, p_loss, real_pca_map, fake_pca_map = train_xcp_orig(batch, label, mask)       
    #print(label, img_name, idx_np)

    write_tfboard([c_loss[0]], itr, name='Class')
    write_tfboard([p_loss[0]], itr, name='Map')

    # ----------------EVAL()--------------------
    if itr % 100 == 0:
        print("Eval: " + str(itr))
        test_results = [0,0]
        for i in range(5):
            batch_real, real_name, lb_real, mask_real = next(testing_batch_real_generator)
            batch_fake, fake_name, lb_fake, mask_fake = next(testing_batch_fake_generator)
            batch = torch.cat((batch_real, batch_fake), 0)
            label = torch.cat((lb_real, lb_fake), 0)

            a, b = test_orig(batch, label)
            test_results[0] += a
            test_results[1] += b

        test_results[0] /= 5
        test_results[1] /= 5
        write_tfboard(test_results, itr, name='TEST')
        # scipy.io.savemat(opt.savedir + '/result/data/'+str(itr)+'.mat', {'mask':mask.data.cpu().numpy(), 'real_map':real_pca_map.data.cpu().numpy(), 'fake_map': fake_pca_map.data.cpu().numpy(), 'img_name':img_name})

    if (itr < 100000) and (itr % 10000 == 0):
        torch.save(model_xcp, '%s/modules/%s/%d.pth' % (opt.savedir, sig, itr))

    if (itr > 100000) and (itr % 1000 == 0):
        torch.save(model_xcp, '%s/modules/%s/%d.pth' % (opt.savedir, sig, itr))

    itr += 1