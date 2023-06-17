import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as wn
from model.Conv4d import Conv4d
from model.swin import Swin


class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()
        self.hatten_l = HistEQatten()
        self.hatten_d = HistEQatten()
        self.relu = nn.LeakyReLU(inplace=True)

        self.swin_l = Swin(dim=32,
                            input_resolution=(25, 128, 128),
                            num_heads=4,
                            window_size=[2, 8, 8],
                            mut_attn=True,
                            depth=5,
                            mlp_ratio=2.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_path=0.,
                            norm_layer=nn.LayerNorm,
                            use_checkpoint_attn=False,
                            use_checkpoint_ffn=False)
        self.swin_d = Swin(dim=32,
                            input_resolution=(25, 128, 128),
                            num_heads=4,
                            window_size=[2, 8, 8],
                            mut_attn=True,
                            depth=5,
                            mlp_ratio=2.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_path=0.,
                            norm_layer=nn.LayerNorm,
                            use_checkpoint_attn=False,
                            use_checkpoint_ffn=False).cuda()
        self.swin_l1 = Swin(dim=32,
                             input_resolution=(25, 128, 128),
                             num_heads=4,
                             window_size=[2, 8, 8],
                             mut_attn=True,
                             depth=5,
                             mlp_ratio=2.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_path=0.,
                             norm_layer=nn.LayerNorm,
                             use_checkpoint_attn=False,
                             use_checkpoint_ffn=False)
        self.swin_d1 = Swin(dim=32,
                             input_resolution=(25, 128, 128),
                             num_heads=4,
                             window_size=[2, 8, 8],
                             mut_attn=True,
                             depth=5,
                             mlp_ratio=2.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_path=0.,
                             norm_layer=nn.LayerNorm,
                             use_checkpoint_attn=False,
                             use_checkpoint_ffn=False).cuda()
        self.swin_l2 = Swin(dim=32,
                             input_resolution=(25, 128, 128),
                             num_heads=4,
                             window_size=[2, 8, 8],
                             mut_attn=True,
                             depth=5,
                             mlp_ratio=2.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_path=0.,
                             norm_layer=nn.LayerNorm,
                             use_checkpoint_attn=False,
                             use_checkpoint_ffn=False)
        self.swin_d2 = Swin(dim=32,
                             input_resolution=(25, 128, 128),
                             num_heads=4,
                             window_size=[2, 8, 8],
                             mut_attn=True,
                             depth=5,
                             mlp_ratio=2.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_path=0.,
                             norm_layer=nn.LayerNorm,
                             use_checkpoint_attn=False,
                             use_checkpoint_ffn=False).cuda()
        self.swin_l3 = Swin(dim=32,
                             input_resolution=(25, 128, 128),
                             num_heads=4,
                             window_size=[2, 8, 8],
                             mut_attn=True,
                             depth=5,
                             mlp_ratio=2.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_path=0.,
                             norm_layer=nn.LayerNorm,
                             use_checkpoint_attn=False,
                             use_checkpoint_ffn=False)
        self.swin_d3 = Swin(dim=32,
                             input_resolution=(25, 128, 128),
                             num_heads=4,
                             window_size=[2, 8, 8],
                             mut_attn=True,
                             depth=5,
                             mlp_ratio=2.,
                             qkv_bias=True,
                             qk_scale=None,
                             drop_path=0.,
                             norm_layer=nn.LayerNorm,
                             use_checkpoint_attn=False,
                             use_checkpoint_ffn=False).cuda()

        self.RDB = RDB(nChannels=64, growthRate=16, nDenselayer=4)
        self.fuse = Conv4d(64, 1, 3, 1)

        self.decom = Conv4d(1, 32, 3, 1)

        self.decoml1 = nn.Conv3d(32, 32, 3, padding=1)
        self.decoml2 = nn.Conv3d(32, 32, 3, padding=1)
        self.decoml3 = nn.Conv3d(32, 1, 3, padding=1)

        self.decomd1 = nn.Conv3d(32, 32, 3, padding=1)
        self.decomd2 = nn.Conv3d(32, 32, 3, padding=1)
        self.decomd3 = nn.Conv3d(32, 1, 3, padding=1)



    def forward(self, lf, dslr, lfh, dslrh):

        N, C, A1, A2, H, W = lf.size()
        lf = lf.view(N, C, -1, H, W)
        dslr = dslr.view(N, C, -1, H, W)
        lfh = lfh.view(N, C, -1, H, W)
        dslrh = dslrh.view(N, C, -1, H, W)

        lo = self.hatten_l(lf, lfh)
        do = self.hatten_d(dslr, dslrh)

        lf_s = self.swin_l(lo)
        dslr_s = self.swin_d(do)
        lf_s2 = self.swin_l1(lf_s)
        dslr_s2 = self.swin_d1(dslr_s)
        lf_s3 = self.swin_l2(lf_s2)
        dslr_s3 = self.swin_d2(dslr_s2)
        lf_s4 = self.swin_l3(lf_s3)
        dslr_s4 = self.swin_d3(dslr_s3)

        fuse = self.relu(self.RDB(torch.cat((lf_s4, dslr_s4), 1)))
        fuse = fuse.view(N, -1, 5, 5, H, W)

        fuse = self.fuse(fuse)
        out = torch.tanh(fuse)


        d = self.relu(self.decom(out))
        d = d.view(N, -1, A1 * A2, H, W)

        d_d1 = self.relu(self.decomd1(d))
        d_d11 = self.relu(self.decomd2(d_d1))
        d_d2 = torch.tanh(self.decomd3(d_d11))

        d_l1 = self.relu(self.decoml1(d))
        d_l11 = self.relu(self.decoml2(d_l1))
        d_l2 = torch.tanh(self.decoml3(d_l11))

        return out, d_l2, d_d2


class Channel_Attention(nn.Module):
    def __init__(self, channel, r):
        super(Channel_Attention, self).__init__()
        self.__avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.__max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.__fc = nn.Sequential(
            nn.Conv3d(channel, channel // r, 1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(channel // r, channel, 1, bias=False),
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)
        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)
        y = self.__sigmoid(y1 + y2)
        return x * y


class HistEQatten(nn.Module):
    def __init__(self):
        super(HistEQatten, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv3d(32, 32, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_a0 = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_a1 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.ca = Channel_Attention(32, 2)

    def forward(self, img, imgh):

        N, C, A1, H, W = img.size()
        img = img.view(N, C, -1, H, W)
        imgh = imgh.view(N, C, -1, H, W)
        f = self.conv_0(img)
        f = self.conv_1(f)

        fh = self.conv_0(imgh)
        fh = self.conv_1(fh)

        fa = self.relu(self.conv_a0(torch.cat((f, fh), 1)))

        fa = self.conv_a1(fa)

        out = f * fa
        out = self.ca(out)

        return out


def conv_layer(inc, outc, kernel_size=3, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='kaiming',
               fan_type='fan_in', activation=False, weight_normalization=True):
    layers = []

    if bn:
        m = nn.BatchNorm3d(inc)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        layers.append(m)

    if activation == 'before':
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

    m = nn.Conv3d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                  groups=groups, bias=bias, stride=1)
    init_gain = 0.02
    if init_type == 'normal':
        torch.nn.init.normal_(m.weight, 0.0, init_gain)
    elif init_type == 'xavier':
        torch.nn.init.xavier_normal_(m.weight, gain=init_gain)
    elif init_type == 'kaiming':
        torch.nn.init.kaiming_normal_(m.weight, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
    elif init_type == 'orthogonal':
        torch.nn.init.orthogonal_(m.weight)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)

    if activation == 'after':
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))

    return nn.Sequential(*layers)


class make_dense(nn.Module):

    def __init__(self, nChannels=64, growthRate=32, pos=False):
        super(make_dense, self).__init__()

        kernel_size = 3
        if pos == 'first':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False,
                                   negative_slope=0.2, bn=True, init_type='kaiming', fan_type='fan_in',
                                   activation=False, weight_normalization=True)
        elif pos == 'middle':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False,
                                   negative_slope=0.2, bn=True, init_type='kaiming', fan_type='fan_in',
                                   activation='before', weight_normalization=True)
        elif pos == 'last':
            self.conv = conv_layer(nChannels, growthRate, kernel_size=kernel_size, groups=1, bias=False,
                                   negative_slope=1, bn=True, init_type='kaiming', fan_type='fan_in',
                                   activation='before', weight_normalization=True)
        else:
            raise NotImplementedError('ReLU position error in make_dense')

    def forward(self, x):
        return torch.cat((x, self.conv(x)), 1)


class RDB(nn.Module):
    def __init__(self, nChannels=96, nDenselayer=5, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []

        modules.append(make_dense(nChannels_, growthRate, 'first'))
        nChannels_ += growthRate
        for i in range(nDenselayer - 2):
            modules.append(make_dense(nChannels_, growthRate, 'middle'))
            nChannels_ += growthRate
        modules.append(make_dense(nChannels_, growthRate, 'last'))
        nChannels_ += growthRate

        self.dense_layers = nn.Sequential(*modules)

        self.conv_1x1 = conv_layer(nChannels_, nChannels, kernel_size=1, groups=1, bias=False, negative_slope=1,
                                   bn=True, init_type='kaiming', fan_type='fan_in', activation=False,
                                   weight_normalization=True)

    def forward(self, x):
        return self.conv_1x1(self.dense_layers(x)) + x

