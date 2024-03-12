from __future__ import print_function, division

import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch

from evaluate_stereo import *
import core.stereo_datasets as datasets
from loader.loader_mvsec import *
import itertools

from utils.visualization import visualize_example



DEVICE = 'cuda'




def demo(args):
    model = EGEIStereo(args) 
    model.load_state_dict(torch.load(args.restore_ckpt))
    logging.info(f"Done loading checkpoint")

    model = model
    model.to(DEVICE)
    model.eval()

    demo_loader = datasets.fetch_dataloader(args)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        n = args.number_of_examples_to_visualize
        
        for i_batch, batch in enumerate(tqdm(itertools.islice(demo_loader,n))):
                example_index = batch['index'].item()
                left_event = batch['event_volume_left'].cuda()
                right_event = batch['event_volume_right'].cuda()
                left_image = batch['left_image'].cuda()
                right_image = batch['right_image'].cuda()

                flow_pre= model(left_event, right_event, left_image, right_image, iters=args.valid_iters)

                flow_pre_last = flow_pre[-1].squeeze()
                visualize_example(batch,example_index,flow_pre_last,output_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--output_directory', help="directory to save output", default="demo_visualization")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=12, help='number of flow-field updates during forward pass')


    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    
    parser.add_argument('-p', '--path', type=str, help="Dataset path", required=True)
    parser.add_argument('-s', '--split', default=1, type=int, help="Set of Indoor Flying splits")
    parser.add_argument('-n', '--num_workers', default=0, type=int, help='How many sub-processes to use for data loading')
    parser.add_argument('--mode', default='train', type=str, help="Mode([train/test/demo)")
    parser.add_argument('--number_of_examples_to_visualize', type=int, default=10, help='the number of examples you want to visulize')

    args = parser.parse_args()

    demo(args)
