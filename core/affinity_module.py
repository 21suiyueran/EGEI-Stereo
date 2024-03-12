from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.nn import init as init
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        # Layer normalization
        self.param_free_norm = nn.LayerNorm([norm_nc,72,88], elementwise_affine=False)

        self.nhidden = nhidden
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = nhidden
        ks = 3
        pw = ks // 2
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.relu = nn.ReLU()

        # print('#########sdf####',norm_nc, label_nc, nhidden)
        self.conv2 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=1,dilation=1)
        self.conv3 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=2,dilation=2)
        self.conv4 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=3,dilation=3)
        self.conv5 = nn.Conv2d(nhidden,nhidden//4,kernel_size=3,padding=4,dilation=4)

        self.final_conv = nn.Conv2d(norm_nc,norm_nc,kernel_size=3,padding=1)



    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        # print("x,segmap",x.size(),segmap.size())
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[-2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        
        img_features_2 = self.conv2(actv)
        img_features_3 = self.conv3(actv)
        img_features_4 = self.conv4(actv)
        img_features_5 = self.conv5(actv)
        
        actv = torch.cat((img_features_2,img_features_3,img_features_4,img_features_5),dim=1)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        # print("normalized,gamma,beta",normalized.size(),gamma.size(),beta.size())
        out = normalized * (1 + gamma) + beta

        return out
    

class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
    

class SpatialAttn(nn.Module):

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        return x

    
class ChannelAttn(nn.Module):

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class EdgeEmbedBlock(nn.Module):
    def __init__(self, high_in_plane, low_in_plane, out_plane):
        super(EdgeEmbedBlock, self).__init__()
        self.conv3x3 = ConvBlock(high_in_plane, out_plane, 3, 1, 1)
        self.conv1x1 = ConvBlock(low_in_plane, out_plane, 1)
        self.conv = ConvBlock(out_plane, out_plane, 1)

    def forward(self, high_x, low_x):
        high_x = F.interpolate(high_x, scale_factor=0.25, mode='nearest')
        high_x = self.conv3x3(high_x)
        low_x = self.conv1x1(low_x)

        return self.conv(high_x * low_x)


class EGEIF_AttentionTransformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EGEIF_AttentionTransformer, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = SPADE(dim,1)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event, edge):

        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event,edge)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused
    

class EGMAAttn(nn.Module):

    def __init__(self, in_channels):
        super(EGMAAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)
        self.norm = nn.Sigmoid()

    def forward(self, x, edge):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        weight = self.norm(edge)
        y = torch.relu(self.conv(y*weight))
        return y