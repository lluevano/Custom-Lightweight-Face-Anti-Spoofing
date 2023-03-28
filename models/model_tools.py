'''MIT License
Copyright (C) 2020 Prokofiev Kirill
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class DYShiftMax(nn.Module):
    def __init__(self, inp=None, oup=None, reduction=4, act_max=1.0, act_relu=True, init_a=[0.0, 0.0], init_b=[0.0, 0.0], relu_before_pool=False, g=(0,4), expansion=False):
        super(DYShiftMax, self).__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(
                nn.ReLU(inplace=True) if relu_before_pool == True else nn.Sequential(),
                nn.AdaptiveAvgPool2d(1)
            )

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4
        print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        print('init-a: {}, init-b: {}'.format(init_a, init_b))

        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze, oup*self.exp),
                h_sigmoid()
        )
        if g is None:
            g = 1
        self.g = g[1]
        if self.g !=1  and expansion:
            self.g = inp // self.g
        print('group shuffle: {}, divide group: {}'.format(self.g, expansion))
        self.gc = inp//self.g
        index=torch.Tensor(range(inp)).view(1,inp,1,1)
        print("INDEX SHAPE")
        print(index.shape)
        index=index.view(1,self.g,self.gc,1,1)
        indexgs = torch.split(index, [1, self.g-1], dim=1)
        indexgs = torch.cat((indexgs[1], indexgs[0]), dim=1)
        indexs = torch.split(indexgs, [1, self.gc-1], dim=2)
        indexs = torch.cat((indexs[1], indexs[0]), dim=2)
        self.index = indexs.view(inp).type(torch.LongTensor)
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x
        
        #Adapted for use inside SE block
        if len(x_in.size()) == 2:
            b, c = x_in.size()
            y = x_in
        else:
            b, c, _, _ = x_in.size()
            y = self.avg_pool(x_in).view(b, c)
        
        y = self.fc(y).view(b, self.oup*self.exp, 1, 1)
        y = (y-0.5) * self.act_max

        #n2, c2, h2, w2 = x_out.size() unused
        if len(x_out.size()) == 2:
            x2 = x_out[:,self.index]
        else:
            x2 = x_out[:,self.index,:,:]

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = torch.split(y, self.oup, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out


def get_activation(name="relu", **kwargs):
    print(f"Entered get_activation with {name}")
    activation = nn.Sequential()
    if name.lower() == "relu":
        activation = nn.ReLU6(inplace=True)
    elif name.lower()=="prelu":
        activation = nn.PReLU()
    elif name.lower() == "dyshiftmax":
        activation = DYShiftMax(reduction=8, 
                     act_max=2.0, act_relu=True, init_a=[1.0, 1.0], init_b=[0.0, 0.0], relu_before_pool=False, expansion=False, **kwargs)
    return activation
        


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0):

        super().__init__()
        self.theta = theta
        self.bias = bias or None
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        if self.groups > 1:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels//in_channels, kernel_size))
        else:
            self.weight = nn.Parameter(kaiming_init(out_channels, in_channels, kernel_size))
        self.padding = padding
        self.i = 0

    def forward(self, x):
        out_normal = F.conv2d(input=x, weight=self.weight, bias=self.bias, dilation=self.dilation,
                              stride=self.stride, padding=self.padding, groups=self.groups)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = self.weight.sum(dim=(2,3), keepdim=True)
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.bias, dilation=self.dilation,
                                stride=self.stride, padding=0, groups=self.groups)
            return out_normal - self.theta * out_diff

def kaiming_init(c_out, c_in, k):
    return torch.randn(c_out, c_in, k, k)*math.sqrt(2./c_in)


class Dropout(nn.Module):
    DISTRIBUTIONS = ['bernoulli', 'gaussian', 'none']

    def __init__(self, p=0.5, mu=0.5, sigma=0.3, dist='bernoulli', linear=False):
        super().__init__()

        self.dist = dist
        assert self.dist in Dropout.DISTRIBUTIONS

        self.p = float(p)
        assert 0. <= self.p <= 1.

        self.mu = float(mu)
        self.sigma = float(sigma)
        assert self.sigma > 0.
        # need to distinct 2d and 1d dropout
        self.linear = linear
    def forward(self, x):
        if self.dist == 'bernoulli' and not self.linear:
            out = F.dropout2d(x, self.p, self.training)
        elif self.dist == 'bernoulli' and self.linear:
            out = F.dropout(x, self.p, self.training)
        elif self.dist == 'gaussian':
            if self.training:
                with torch.no_grad():
                    soft_mask = x.new_empty(x.size()).normal_(self.mu, self.sigma).clamp_(0., 1.)

                scale = 1. / self.mu
                out = scale * soft_mask * x
            else:
                out = x
        else:
            out = x

        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        #self.prelu = nn.PReLU() #unused
        
    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, activation="PReLU", **act_kwargs):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, make_divisible(channel // reduction, 8)),
                #nn.ReLU(inplace=True),
                get_activation(name="prelu", **(dict(inp=make_divisible(channel // reduction, 8), oup=make_divisible(channel // reduction, 8)))),
                nn.Linear(make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_in(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.InstanceNorm2d(oup),
        h_swish()
    )

def conv_3x3_bn(inp, oup, stride, theta):
    return nn.Sequential(
        Conv2d_cd(inp, oup, 3, stride, 1, bias=False, theta=theta),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_in(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.InstanceNorm2d(oup),
        h_swish()
    )

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class AntiSpoofModel(nn.Module):
    """parent class for mobilenets"""
    def __init__(self, width_mult, prob_dropout, type_dropout,
                 prob_dropout_linear, embeding_dim, mu, sigma,
                 theta, multi_heads):
        super().__init__()
        self.prob_dropout = prob_dropout
        self.type_dropout = type_dropout
        self.width_mult = width_mult
        self.prob_dropout_linear = prob_dropout_linear
        self.embeding_dim = embeding_dim
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.multi_heads = multi_heads[0]
        self.multi_spoof = multi_heads[1]
        self.features = nn.Identity()
        self.include_spoofer=False

        # building last several layers
        self.conv_last = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.spoofer = nn.Linear(embeding_dim, 2)
        if self.multi_heads:
            self.lightning = nn.Linear(embeding_dim, 5)
            self.spoof_type = nn.Linear(embeding_dim, 11)
            self.real_atr = nn.Linear(embeding_dim, 40)
        elif self.multi_spoof:
            self.spoof_type = nn.Linear(embeding_dim, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        if self.include_spoofer:
            x = x.view(x.size(0), -1)
            x = self.spoofer(x)
        return x

    def make_logits(self, features, all=False):
        all = all if self.multi_heads or self.multi_spoof else False
        output = features.view(features.size(0), -1)
        spoof_out = self.spoofer(output)
        if all and self.multi_heads:
            type_spoof = self.spoof_type(output)
            lightning_type = self.lightning(output)
            real_atr = torch.sigmoid(self.real_atr(output))
            return spoof_out, type_spoof, lightning_type, real_atr
        
        elif all and self.multi_spoof:
            type_spoof = self.spoof_type(output)
            return spoof_out, type_spoof
        
        return spoof_out

    def forward_to_onnx(self,x, scaling=1):
        x = self.features(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        spoof_out = self.spoofer(x)
        if isinstance(spoof_out, tuple):
            spoof_out = spoof_out[0]
        probab = F.softmax(spoof_out*scaling, dim=-1)
        return probab
