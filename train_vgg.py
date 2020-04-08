import argparse
import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from imageio import imread
from network.vgg import vgg16
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
torch.backends.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='manual seed')
parser.add_argument('--it_start', type=int, default=1, help='number of itr to start with')
parser.add_argument('--it_end', type=int, default=10000, help='number of itr to end with')
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
    def __init__(self, data_root, image_width=224, image_height=224, seed=opt.seed):

        np.random.seed(seed)
        self.data_root = data_root
        self.image_width = image_width
        self.image_height = image_height

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_width, image_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = {'Real': 0, 'Fake': 1}
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
        random_folder = random.choice(list(self.classes.keys()))
        label = self.classes[random_folder]

        image_path = os.path.join(self.data_root, random.choice(self.img_paths[random_folder])[0])
        img = self.load_img(image_path)

        return img, label

    def __len__(self):
        return sum([len(i) for i in self.img_paths.values()])


def get_training_batch(data_loader):
    while True:
        for sequence in data_loader:
            batch = sequence[0].cuda(), sequence[1].cuda()
            yield batch


print("Initializing Data Loader")
train_data = DATA(data_root=(opt.data_dir + 'train/'))
train_loader = DataLoader(train_data, num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory=True)
training_batch_generator = get_training_batch(train_loader)

test_data = DATA(data_root=(opt.data_dir + 'validation/'))
test_loader = DataLoader(test_data, num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True, pin_memory=True)
testing_batch_generator = get_training_batch(test_loader)


print("Initializing Networks")
model_vgg = vgg16(pretrained=True, progress=True)
optimizer_vgg = optim.Adam(model_vgg.parameters(), lr=opt.lr)
model_vgg.cuda()
cse_loss = nn.CrossEntropyLoss().cuda()


def train(batch, label):
    model_vgg.train()
    y = model_vgg(batch)
    loss = cse_loss(y, label)
    optimizer_vgg.zero_grad()
    loss.backward()
    optimizer_vgg.step()
    return[loss.item()]


def test(batch, label):
    model_vgg.eval()
    with torch.no_grad():
        n = model_vgg(batch)
        loss = cse_loss(n, label)
        prediction = torch.max(n, dim=1)[1]
        accu = (prediction.eq(label.long())).sum()
    return [loss.item(), accu.item()/len(batch)]


def write_tfboard(vals, itr, name):
    for idx, item in enumerate(vals):
        writer.add_scalar('data/%s%d' % (name, idx), item, itr)


writer = SummaryWriter('%s/logs/%s' % (opt.save_dir, sig))

print("Start Training")
itr = opt.it_start
while itr != opt.it_end+1:
    batch_train, lb_train = next(training_batch_generator)
    loss = train(batch_train, lb_train)
    write_tfboard([loss[0]], itr, name='TRAIN')

    if itr % 100 == 0:
        test_results = [0,0]
        for i in range(5):
            batch_test, lb_test = next(testing_batch_generator)
            a, b = test(batch_test, lb_test)
            test_results[0] += a
            test_results[1] += b
        test_results[0] /= 5
        test_results[1] /= 5
        write_tfboard(test_results, itr, name='TEST')
        print("Eval: " + str(itr))
    if itr % 1000 == 0:
        torch.save({'module': model_vgg.state_dict()}, '%s/modules/%s/%d.pickle' % (opt.save_dir, sig, itr))
        print("Save Model: {:d}".format(itr))

    itr += 1
