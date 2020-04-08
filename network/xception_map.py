import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from torch.autograd import Variable
import numpy as np


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, templates, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)
        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        # map related ops
        self.templates = templates
        self.sigmoid = nn.Sigmoid().cuda()
        self.map = SeparableConv2d(728, 1, 3, stride=1, padding=1, bias=False)
        self.bnmap = nn.BatchNorm2d(728)
        self.convmap = SeparableConv2d(728, 10, 3, 1, 1)
        self.convmap2 = SeparableConv2d(10, 1, 3, 1, 1)
        self.linearmap = nn.Linear(3610, 10)
        self.softmax = torch.nn.Softmax(dim=1)
        self.avgpool2d = nn.AvgPool2d(19, stride=1)

        self.map_conv1=Block(728,364,2,2,start_with_relu=True,grow_first=False)
        self.map_linear = nn.Linear(364, 10)

    def mask_reg(self, x):
        mask = self.map(x)
        mask = self.sigmoid(mask)
        x = x * mask
        return x, mask, 0

    def mask_template(self, x):
        vec = self.map_conv1(x)
        vec = self.relu(vec)
        vec = F.adaptive_avg_pool2d(vec, (1, 1))
        vec = vec.view(vec.size(0), -1)
        vec = self.map_linear(vec)
        mask = torch.mm(vec, self.templates.reshape(10, 361))
        mask = mask.reshape(x.shape[0], 1, 19, 19)
        x = x * mask
        return x, mask, vec

    def mask_pca_template(self, x):
        fe = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        fe = torch.transpose(fe, 1, 2)
        mu = torch.mean(fe, 2, keepdim=True)
        fea_diff = fe - mu

        cov_fea = torch.bmm(fea_diff, torch.transpose(fea_diff, 1, 2))
        B = self.templates.reshape(1, 10, 361).repeat(x.shape[0], 1, 1)
        D = torch.bmm(torch.bmm(B, cov_fea), torch.transpose(B, 1, 2))
        eigen_value, eigen_vector = D.symeig(eigenvectors=True)
        index = torch.tensor([9]).cuda()
        eigen = torch.index_select(eigen_vector, 2, index)

        vec = eigen.squeeze(-1)
        mask = torch.mm(vec, self.templates.reshape(10, 361))
        mask = mask.reshape(x.shape[0], 1, 19, 19)
        x = x * mask
        return x, mask, vec

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        # x, mask, vec = self.mask_template(x)
        x, mask, vec = self.mask_reg(x)
        # x, mask, vec = self.mask_pca_template(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x, mask, vec
        # return x, 0, 0

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x, mask, vec = self.features(input)
        x = self.logits(x)
        return x, mask, vec


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('SeparableConv2d') != -1:
        m.conv1.weight.data.normal_(0.0, 0.01)
        if m.conv1.bias is not None:
            m.conv1.bias.data.fill_(0)
        m.pointwise.weight.data.normal_(0.0, 0.01)
        if m.pointwise.bias is not None:
            m.pointwise.bias.data.fill_(0)
    elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('LSTM') != -1:
        for i in m._parameters:
            if i.__class__.__name__.find('weight') != -1:
                i.data.normal_(0.0, 0.01)
            elif i.__class__.__name__.find('bias') != -1:
                i.bias.data.fill_(0)


def xception(templates=0, num_classes=2, load_pretrain=True):
    model = Xception(templates, num_classes=num_classes)
    if load_pretrain:
        state_dict = torch.load('./network/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict, False)
    else:
        model.apply(init_weights)
    model.last_linear = nn.Linear(2048, num_classes)
    return model
