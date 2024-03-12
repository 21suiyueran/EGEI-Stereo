from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.egei_stereo import EGEIStereo, autocast
import core.stereo_datasets as datasets
from utils import errors


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate(model, vali_loader, dataset, iters=32, mixed_prec=False):

    model.eval()
     
    one_pixel_error_list, mean_disparity_error_list, mean_depth_error_list, median_depth_error_list = [],[],[],[]
    for index, batch in enumerate(tqdm(vali_loader)):

        left_event = batch['event_volume_left'].cuda()
        right_event = batch['event_volume_right'].cuda()
        left_image = batch['left_image'].cuda()
        right_image = batch['right_image'].cuda()
        disparity_image = batch["disparity_image"].cuda()


        with autocast(enabled = mixed_prec):
            flow_preds= model(left_event, right_event, left_image, right_image, iters=iters)

        estimated_depth = dataset.disparity_to_depth(flow_preds[-1])
        ground_truth_depth = dataset.disparity_to_depth(disparity_image)
        binary_error_map, one_pixel_error = errors.compute_n_pixels_error(flow_preds[-1], disparity_image, n=1.0)
        mean_disparity_error = errors.compute_absolute_error(flow_preds[-1], disparity_image)[1]
        mean_depth_error = errors.compute_absolute_error(estimated_depth, ground_truth_depth)[1]
        median_depth_error = errors.compute_absolute_error(estimated_depth, ground_truth_depth,use_mean=False)[1]

        one_pixel_error_list.append(one_pixel_error)
        mean_disparity_error_list.append(mean_disparity_error)
        mean_depth_error_list.append(mean_depth_error)
        median_depth_error_list.append(median_depth_error)



    one_pixel_error_list = np.array(one_pixel_error_list)
    mean_disparity_error_list = np.array(mean_disparity_error_list)
    mean_depth_error_list = np.array(mean_depth_error_list)
    median_depth_error_list = np.array(median_depth_error_list)

    one_pixel_error = np.mean(one_pixel_error_list)
    mean_disparity_error = np.mean(mean_disparity_error_list)
    mean_depth_error = np.mean(mean_depth_error_list)
    median_depth_error = np.mean(median_depth_error_list)

    print("one_pixel_accuracy: %f, mean_disparity_error: %f, mean_depth_error: %f, median_depth_error: %f" % (100 - one_pixel_error, mean_disparity_error, mean_depth_error, median_depth_error))
    return {'one_pixel_accuracy_vali': 100 - one_pixel_error, 'mean_disparity_error_vali': mean_disparity_error,'mean_depth_error_vali':mean_depth_error,'median_depth_error_vali':median_depth_error}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=12, help='number of flow-field updates during forward pass')

    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")

    parser.add_argument('-s', '--split', default=1, type=int, help="Set of Indoor Flying splits")
    parser.add_argument('-p', '--path', type=str, help="Dataset path", required=True)
    parser.add_argument('-t', '--type', default='warm_start', type=str, help="Evaluation type ([warm_start]/standard)")
    parser.add_argument('-n', '--num_workers', default=1, type=int, help='How many sub-processes to use for data loading')
    parser.add_argument('--mode', default='test', type=str, help="Mode([train/test/demo)")

    args = parser.parse_args()

    test_loader, test_set, vali_test_set = datasets.fetch_dataloader(args)
    print('Testing with %d image pairs' % len(test_set))

    model = EGEIStereo(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")


    validate(model, test_loader, vali_test_set, iters=args.valid_iters)
