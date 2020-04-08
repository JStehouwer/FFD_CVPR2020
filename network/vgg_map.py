
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import load_state_dict_from_url
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

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


class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self, templates, num_classes=2):
        """
        Constructor
        """
        super(VGG_16, self).__init__()

        self.templates = templates
        self.map_conv1=Block(256,128,2,2,start_with_relu=True,grow_first=False)
        self.map_linear = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            )
        self._initialize_weights()

    def mask_template(self, x):
        vec = self.map_conv1(x)
        vec = self.relu(vec)
        vec = F.adaptive_avg_pool2d(vec, (1, 1))
        vec = vec.view(vec.size(0), -1)
        vec = self.map_linear(vec)
        mask = torch.mm(vec, self.templates.reshape(10, 784))
        mask = mask.reshape(x.shape[0], 1, 28, 28)
        x = x * mask
        return x, mask, vec

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (224x224)

        Returns: class logits

        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)      # B*64*112*112
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)      # B*128*56*56
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)      # B*256*28*28
        x, mask, vec = self.mask_template(x)

        # map_ = torch.sigmoid(x[:,-1,:,:].unsqueeze(1))
        # x = x[:,0:-1,:,:]
        # x = map_.expand_as(x).mul(x)

        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)      # B*512*14*14
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))

        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # input()
        x = self.classifier(x)

        return x, mask, vec
        # return x, 0, 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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


def vgg16(templates=0, num_classes=2, load_pretrain=True, progress=True, **kwargs):
    model = VGG_16(templates, num_classes)

    # model = torch.load('./pretrain_vgg16.pth')
    # print("loaded")
    if load_pretrain:
        state_dict = load_state_dict_from_url(model_urls['vgg16'],
                                              progress=progress)
        state_dict_new = {}
        for name, weights in state_dict.items():
            # print(name)
            if 'features' in name:
                state_dict_new['conv_1_1.weight'] = state_dict['features.0.weight']
                state_dict_new['conv_1_2.weight'] = state_dict['features.2.weight']
                state_dict_new['conv_2_1.weight'] = state_dict['features.5.weight']
                state_dict_new['conv_2_2.weight'] = state_dict['features.7.weight']
                state_dict_new['conv_3_1.weight'] = state_dict['features.10.weight']
                state_dict_new['conv_3_2.weight'] = state_dict['features.12.weight']
                state_dict_new['conv_3_3.weight'] = state_dict['features.14.weight']
                state_dict_new['conv_4_1.weight'] = state_dict['features.17.weight']
                state_dict_new['conv_4_2.weight'] = state_dict['features.19.weight']
                state_dict_new['conv_4_3.weight'] = state_dict['features.21.weight']
                state_dict_new['conv_5_1.weight'] = state_dict['features.24.weight']
                state_dict_new['conv_5_2.weight'] = state_dict['features.26.weight']
                state_dict_new['conv_5_3.weight'] = state_dict['features.28.weight']
                state_dict_new['conv_1_1.bias'] = state_dict['features.0.bias']
                state_dict_new['conv_1_2.bias'] = state_dict['features.2.bias']
                state_dict_new['conv_2_1.bias'] = state_dict['features.5.bias']
                state_dict_new['conv_2_2.bias'] = state_dict['features.7.bias']
                state_dict_new['conv_3_1.bias'] = state_dict['features.10.bias']
                state_dict_new['conv_3_2.bias'] = state_dict['features.12.bias']
                state_dict_new['conv_3_3.bias'] = state_dict['features.14.bias']
                state_dict_new['conv_4_1.bias'] = state_dict['features.17.bias']
                state_dict_new['conv_4_2.bias'] = state_dict['features.19.bias']
                state_dict_new['conv_4_3.bias'] = state_dict['features.21.bias']
                state_dict_new['conv_5_1.bias'] = state_dict['features.24.bias']
                state_dict_new['conv_5_2.bias'] = state_dict['features.26.bias']
                state_dict_new['conv_5_3.bias'] = state_dict['features.28.bias']
            else:
                state_dict_new[name] = state_dict[name]
        del state_dict_new['classifier.6.weight']
        del state_dict_new['classifier.6.bias']
        del state_dict_new['conv_3_3.weight']
        del state_dict_new['conv_3_3.bias']
        model.load_state_dict(state_dict_new, False)
        # model = torch.load('./pretrain_vgg16.pth')
        # print("loaded")
    else:
        model.apply(init_weights)
    return model