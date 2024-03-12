import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicUpdateBlock
from core.extractor import BasicEncoder
from core.corr import CorrBlock1D
from utils.utils import coords_grid, InputPadder
from core import  perturbations
from core.affinity_module import *
from core.ted import TED 

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

        
def argtopk(x, axis=-1):
    _, index = torch.topk(x, k=3, dim=axis)  

    return F.one_hot(index, list(x.shape)[axis]).float()


class EGEIStereo(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.cnet = BasicEncoder(output_dim=256, norm_fn=args.context_norm, downsample=args.n_downsample)

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=128)

        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance', downsample=args.n_downsample)

        self.conv_score = nn.Sequential(convbn(8, 64, 3, 2, 1, 1), 
                                       nn.ReLU(inplace=True),
                                       convbn(64, 128, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 128, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(64, 5, 3, 1, 1, 1), # 5->15
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d(1))

        self.conv_score = nn.Sequential(*self.conv_score)

        self.pert_argtopk = perturbations.perturbed(argtopk, 
                                        num_samples=200, 
                                        sigma=0.05, 
                                        noise='gumbel',
                                        batched=True)
        
        self.fuse = EGEIF_AttentionTransformer(128, num_heads=1, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')

        ed = TED()
        ed.load_state_dict(torch.load('pre-trained/TEED_model.pth'), strict=True)
        self.ed = ed

        self.edgeembed = EdgeEmbedBlock(1, 128, 128)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, left_event, right_event, left_image, right_image, iters=12):
        """ Estimate disparity between pair of frames """

        padder = InputPadder(left_event.shape, divis_by=32)
        left_event, right_event = padder.pad(left_event, right_event)
        left_image, right_image = padder.pad(left_image,right_image)


        b, c, h, w = left_event.shape
        left_score = self.conv_score(torch.cat((left_event, left_image), 1)).squeeze(-1).squeeze(-1)
        right_score = self.conv_score(torch.cat((right_event, right_image), 1)).squeeze(-1).squeeze(-1)

        left_one_hot = self.pert_argtopk(left_score) 
        right_one_hot = self.pert_argtopk(right_score) 

        left_event = torch.bmm(left_one_hot, left_event.view(b, c, -1)).view(b, -1, h, w)
        right_event = torch.bmm(right_one_hot, right_event.view(b, c, -1)).view(b, -1, h, w)

        event1 = right_event.contiguous()
        event2 = left_event.contiguous()

        image1 = (2 * (right_image / 255.0) - 1.0).contiguous()
        image2 = (2 * (left_image / 255.0) - 1.0).contiguous()

        edge1 = self.ed(right_image)
        edge2 = self.ed(left_image)

        with autocast(enabled=self.args.mixed_precision):


            cnet = self.cnet(image2)

            eventfmap1, eventfmap2 = self.fnet([event1, event2])
            imagefmap1, imagefmap2 = self.fnet([image1, image2])

            fmap1 = self.fuse(imagefmap1, eventfmap1, edge1)
            fmap2 = self.fuse(imagefmap2, eventfmap2, edge2)
            fmap1, fmap2 = fmap1.float(), fmap2.float()

            net, inp = torch.split(cnet,[128,128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)


        corr_block = CorrBlock1D
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        
        coords0, coords1 = self.initialize_flow(net)

        disp_predictions = []

        for itr in range(iters):
            coords1 = coords1.detach()

            corr = corr_fn(coords1)

            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                # Rather than embedding edge into context features multiple times during the update process, we do it once at the beginning 
                embedded = self.edgeembed(edge2, inp)
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, embedded)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            coords1 = coords1 + delta_flow

            flow_up = self.upsample_flow(coords1 - coords0, up_mask)


            disp_up = flow_up[:,:1]

            disp_up = padder.unpad(disp_up)
            
            disp_predictions.append(disp_up)

        return disp_predictions
