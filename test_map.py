import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from imageio import imread
from network.xception_map import xception
from network.vgg_map import vgg16
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, auc
import matplotlib.pyplot as plt
import pickle
from scipy.io import savemat
torch.backends.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--data_dir', help='root directory for data')
parser.add_argument('--modeldir', help='model in pickle file to test')
parser.add_argument('--network', default='xcp', help='directory for result')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
data_dir = opt.data_dir + 'test/'
#################################################################################################################
class DATA(object):
    def __init__(self, data_root, image_width=299, image_height=299):
        self.data_root = data_root
        self.image_width = image_width
        self.image_height = image_height

        transform_xcp = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((299, 299)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5] * 3, [0.5] * 3)])

        transform_vgg = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


        if opt.network == 'xcp':
            self.transform = transform_xcp
        if opt.network == 'vgg':
            self.transform = transform_vgg

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


def display(mask_gt, mask_pred):
    mask_gt_cpu = mask_gt.cpu()
    mask_gt_concat = np.concatenate((mask_gt_cpu[0], mask_gt_cpu[1], mask_gt_cpu[2], mask_gt_cpu[3],
                                     mask_gt_cpu[4], mask_gt_cpu[5], mask_gt_cpu[6], mask_gt_cpu[7],
                                     mask_gt_cpu[8], mask_gt_cpu[9], mask_gt_cpu[10], mask_gt_cpu[11],
                                     mask_gt_cpu[12], mask_gt_cpu[13], mask_gt_cpu[14], mask_gt_cpu[15]), axis=2)

    mask_bin_cpu = torch.where(mask_gt_cpu < 0.1,  torch.zeros(19, 19), torch.ones(19, 19))
    mask_bin_concat = np.concatenate((mask_bin_cpu[0], mask_bin_cpu[1], mask_bin_cpu[2], mask_bin_cpu[3],
                                      mask_bin_cpu[4], mask_bin_cpu[5], mask_bin_cpu[6], mask_bin_cpu[7],
                                      mask_bin_cpu[8], mask_bin_cpu[9], mask_bin_cpu[10], mask_bin_cpu[11],
                                      mask_bin_cpu[12], mask_bin_cpu[13], mask_bin_cpu[14], mask_bin_cpu[15]), axis=2)

    vec_gt = torch.from_numpy(np.dot(mask_bin_cpu.reshape(16, 361).numpy(), np.linalg.pinv(templates.cpu().reshape(10, 361).numpy()))).cuda()
    mask_calc_cpu = torch.mm(vec_gt, templates.reshape(10, 361)).reshape((16, 1, 19, 19)).cpu()
    mask_calc_concat = np.concatenate((mask_calc_cpu[0], mask_calc_cpu[1], mask_calc_cpu[2], mask_calc_cpu[3],
                                       mask_calc_cpu[4], mask_calc_cpu[5], mask_calc_cpu[6], mask_calc_cpu[7],
                                       mask_calc_cpu[8], mask_calc_cpu[9], mask_calc_cpu[10], mask_calc_cpu[11],
                                       mask_calc_cpu[12], mask_calc_cpu[13], mask_calc_cpu[14], mask_calc_cpu[15]), axis=2)

    mask_pred_cpu = mask_pred.cpu().detach()
    mask_pred_concat = np.concatenate((mask_pred_cpu[0], mask_pred_cpu[1], mask_pred_cpu[2], mask_pred_cpu[3],
                                       mask_pred_cpu[4], mask_pred_cpu[5], mask_pred_cpu[6], mask_pred_cpu[7],
                                       mask_pred_cpu[8], mask_pred_cpu[9], mask_pred_cpu[10], mask_pred_cpu[11],
                                       mask_pred_cpu[12], mask_pred_cpu[13], mask_pred_cpu[14], mask_pred_cpu[15]), axis=2)

    mask_out = np.concatenate((mask_gt_concat, mask_bin_concat, mask_calc_concat, mask_pred_concat), axis=1)
    mask_out = np.repeat(mask_out.reshape(76, 304, 1), 3, axis=2)
    mask_out = cv2.resize(mask_out, (1216, 304))
    cv2.imshow('mask_out', mask_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print("Initializing Data Loader")
data = DATA(data_root=(data_dir))
loader = DataLoader(data, num_workers=8, batch_size=opt.batch_size, shuffle=False, drop_last=False, pin_memory=True)
templates = load_template(0)


print("Initializing Networks")
if opt.network == 'xcp':
    model = xception(templates, 2, False)
if opt.network == 'vgg':
    model = vgg16(templates, 2, False)

checkpoint = torch.load(opt.modeldir)
model.load_state_dict(checkpoint['module'])
model.eval().cuda()

sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=1)
def test(batch):
    with torch.no_grad():
        x, mask, vec = model(batch)
        pred = torch.max(x, dim=1)[1]
        x = softmax(x)
    return pred, x, mask


result = {}
fnames = []
scores = []
preds = []
labels = []
total = (len(os.listdir(data_dir + 'Real/')) + len(os.listdir(data_dir + 'Fake/'))) // opt.batch_size
for iteration, batch in enumerate(loader):
    fname, image, label = batch[0], batch[1].cuda(), batch[2].cuda()
    print('\rPrograss: {:d}/{:d}'.format(iteration, total), end='')
    pred, score, map = test(image)
    score_list = score.select(1, 1).tolist()
    pred_list = pred.tolist()
    label_list = label.tolist()
    fnames.extend(fname)
    scores.extend(score_list)
    preds.extend(pred_list)
    labels.extend(label_list)

    for i in range(len(fname)):
        result[fname[i]] = [label_list[i], score_list[i]]
    for i in range(len(fname)):
        vutils.save_image(map[i], './runs/mask_pred/' + fname[i], nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

pickle.dump(result, open('./xcp_mam_result.pickle', 'wb'))
print('\rPrograss: {:d}/{:d}'.format(iteration, total))
acc = accuracy_score(labels, preds)
fpr, tpr, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
fnr = 1 - tpr
eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
tprs = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
for i in range(len(fpr)):
    if fpr[i] > 0.0000000001 and tprs[0] == -1:
        tprs[0] = tpr[i-1]
    if fpr[i] > 0.000000001 and tprs[1] == -1:
        tprs[1] = tpr[i-1]
    if fpr[i] > 0.00000001 and tprs[2] == -1:
        tprs[2] = tpr[i-1]
    if fpr[i] > 0.0000001 and tprs[3] == -1:
        tprs[3] = tpr[i-1]
    if fpr[i] > 0.000001 and tprs[4] == -1:
        tprs[4] = tpr[i-1]
    if fpr[i] > 0.00001 and tprs[5] == -1:
        tprs[5] = tpr[i-1]
    if fpr[i] > 0.0001 and tprs[6] == -1:
        tprs[6] = tpr[i-1]
    if fpr[i] > 0.001 and tprs[7] == -1:
        tprs[7] = tpr[i-1]
    if fpr[i] > 0.01 and tprs[8] == -1:
        tprs[8] = tpr[i-1]
    if fpr[i] > 0.1 and tprs[9] == -1:
        tprs[9] = tpr[i-1]

def calculate_pbca():
    transform_mask_xcp = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((19, 19)),
                                                  transforms.Grayscale(num_output_channels=1),
                                                  transforms.ToTensor()])

    transform_mask_vgg = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((28, 28)),
                                                  transforms.Grayscale(num_output_channels=1),
                                                  transforms.ToTensor()])

    if opt.network == 'xcp':
        transform_mask = transform_mask_xcp
    if opt.network == 'vgg':
        transform_mask = transform_mask_vgg
    mask_dir = "/home/hao/HotData/v4_mask/test/Mask/"
    pred_dir = "/home/hao/Fake-Face-Detection/runs/mask_pred/Fake/"
    mask_name_list = os.listdir(mask_dir)

    mask_list = []
    pred_list = []
    for mask_name in mask_name_list:
        mask = transform_mask(imread(mask_dir + mask_name))
        pred = transform_mask(imread(pred_dir + mask_name))
        print('\rPrograss: {:d}/{:d}'.format(len(mask_list), len(mask_name_list)), end='')
        mask_list.append(mask)
        pred_list.append(pred)

    masks = torch.stack(mask_list).squeeze(1)
    preds = torch.stack(pred_list).squeeze(1)
    pbca = 0.01
    if opt.network == 'xcp':
        masks_bin = torch.where(masks < 0.1, torch.zeros(masks.shape[0], 19, 19), torch.ones(masks.shape[0], 19, 19))
        preds_bin = torch.where(preds < 0.1, torch.zeros(preds.shape[0], 19, 19), torch.ones(preds.shape[0], 19, 19))
        pbca = torch.sum(torch.eq(masks_bin, preds_bin)).float() / (masks_bin.shape[0] * 361)
    if opt.network == 'vgg':
        masks_bin = torch.where(masks < 0.1, torch.zeros(masks.shape[0], 28, 28), torch.ones(masks.shape[0], 28, 28))
        preds_bin = torch.where(preds < 0.1, torch.zeros(preds.shape[0], 28, 28), torch.ones(preds.shape[0], 28, 28))
        pbca = torch.sum(torch.eq(masks_bin, preds_bin)).float() / (masks_bin.shape[0] * 784)
    return pbca


pbca = calculate_pbca()
print("PBCA: {:f}".format(pbca))
roc_auc = auc(fpr, tpr)
result = [fpr, tpr]
plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC fold (AUC = %0.2f)' % (roc_auc))
print("ACC: {:f}\nAUC: {:f}\nEER: {:f}".format(acc, roc_auc, eer))
for i in range(10):
    print("10e-{:d}: {:f}".format(10-i, tprs[i]))
#
# out_file = open('./runs/temp.pickle', 'wb')
# pickle.dump(result, out_file)
# out_file.close()
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
