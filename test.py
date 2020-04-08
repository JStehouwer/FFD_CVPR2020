import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from imageio import imread
from network.xception import xception
from network.vgg import vgg16
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
torch.backends.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--data_dir', help='root directory for data')
parser.add_argument('--save_dir', default='./runs', help='directory for result')
parser.add_argument('--modeldir', help='model in pickle file to test')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

#################################################################################################################
class DATA(object):
    def __init__(self, data_root, image_width=299, image_height=299):
        self.data_root = data_root
        self.image_width = image_width
        self.image_height = image_height

        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((image_width, image_height)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5] * 3, [0.5] * 3)
        # ])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = {'Real': 0, 'Fake': 1}
        self.img_paths = []

        for f in self.classes.items():
            file_names = os.listdir(os.path.join(self.data_root, f[0]))
            for file_name in file_names:
                self.img_paths.append((os.path.join(f[0], file_name), f[1]))

    def __getitem__(self, index):
        fname = self.img_paths[index][0]
        image_path = os.path.join(self.data_root, fname)
        image = imread(image_path)
        image = self.transform(image)
        label = self.img_paths[index][1]
        return fname, image, label

    def __len__(self):
        return len(self.img_paths)


print("Initializing Data Loader")
data = DATA(data_root=(opt.data_dir + 'test/'))
loader = DataLoader(data, num_workers=8, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True)


print("Initializing Networks")
# model_xcp = xception(2)
# checkpoint = torch.load(opt.modeldir)
# model_xcp.load_state_dict(checkpoint['module'])
# model_xcp.eval().cuda()

model_vgg = vgg16()
checkpoint = torch.load(opt.modeldir)
model_vgg.load_state_dict(checkpoint['module'])
model_vgg.eval().cuda()

softmax = nn.Softmax(dim=1)
def test(image):
    with torch.no_grad():
        z = model_vgg(image)
        pred = torch.max(z, dim=1)[1]
        z = softmax(z)
    return pred, z


scores = []
preds = []
labels = []
for iteration, batch in enumerate(loader):
    fname, image, label = batch[0], batch[1].cuda(), batch[2].cuda()
    print("Iteration: " + str(iteration))
    pred, score = test(image)
    scores.extend(score.select(1, 1).tolist())
    preds.extend(pred.tolist())
    labels.extend(label.tolist())

print(scores)
print(preds)
print(labels)
pickle.dump([scores, preds, labels], open( "vgg_base.pickle", "wb" ) )
acc = accuracy_score(labels, preds)
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
print(fpr)
print(tpr)
fnr = 1 - tpr
eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
tpr_0_01 = -1
tpr_0_02 = -1
tpr_0_05 = -1
tpr_0_10 = -1
tpr_0_20 = -1
tpr_0_50 = -1
tpr_1_00 = -1
tpr_2_00 = -1
tpr_5_00 = -1
for i in range(len(fpr)):
    if fpr[i] > 0.0001 and tpr_0_01 == -1:
        tpr_0_01 = tpr[i-1]
    if fpr[i] > 0.0002 and tpr_0_02 == -1:
        tpr_0_02 = tpr[i-1]
    if fpr[i] > 0.0005 and tpr_0_05 == -1:
        tpr_0_05 = tpr[i-1]
    if fpr[i] > 0.001 and tpr_0_10 == -1:
        tpr_0_10 = tpr[i-1]
    if fpr[i] > 0.002 and tpr_0_20 == -1:
        tpr_0_20 = tpr[i-1]
    if fpr[i] > 0.005 and tpr_0_50 == -1:
        tpr_0_50 = tpr[i-1]
    if fpr[i] > 0.01 and tpr_1_00 == -1:
        tpr_1_00 = tpr[i-1]
    if fpr[i] > 0.02 and tpr_2_00 == -1:
        tpr_2_00 = tpr[i-1]
    if fpr[i] > 0.05 and tpr_5_00 == -1:
        tpr_5_00 = tpr[i-1]

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC fold (AUC = %0.2f)' % (roc_auc))
print("ACC: {:f}\nAUC: {:f}\nEER: {:f}\nTPR@0.01: {:f}\nTPR@0.10: {:f}\nTPR@1.00: {:f}".format(acc, roc_auc, eer, tpr_0_01, tpr_0_10, tpr_1_00))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()