from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim

from evaluate_stereo import *
import core.stereo_datasets as datasets
from loader.loader_mvsec import *
from utils import errors
import torch.nn.functional as F

from torch.cuda.amp import GradScaler

####### L1-Loss ################
def sequence_loss(flow_preds, disp_gt, original_dataset, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    valid = (disp_gt != float('inf'))
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        i_weight = loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, flow_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool()].mean() 
       
    

    estimated_depth = original_dataset.disparity_to_depth(flow_preds[-1])
    ground_truth_depth = original_dataset.disparity_to_depth(disp_gt)
    binary_error_map, one_pixel_error = errors.compute_n_pixels_error(flow_preds[-1], disp_gt, n=1.0)
    mean_disparity_error = errors.compute_absolute_error(flow_preds[-1], disp_gt)[1]
    mean_depth_error = errors.compute_absolute_error(estimated_depth, ground_truth_depth)[1]
    median_depth_error = errors.compute_absolute_error(estimated_depth, ground_truth_depth,use_mean=False)[1]

    metrics = {
        'one_pixel_error': one_pixel_error,
        'mean_disparity_error': mean_disparity_error,
        'mean_depth_error': mean_depth_error,
        'median_depth_error': median_depth_error,
    }

    return disp_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, 300000,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def argtopk(x, axis=-1):
    _, index = torch.topk(x, k=3, dim=axis)
  
    return F.one_hot(index, list(x.shape)[axis]).float()


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.epoch = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        self.epoch += 1

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.epoch)

    def close(self):
        self.writer.close()


def train(args):

    scaler = GradScaler(enabled=args.mixed_precision)

    model = EGEIStereo(args)

    print("Parameter Count: %d" % count_parameters(model))

    train_loader,vali_loader, vali_set= datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    model.freeze_bn() # We keep BatchNorm frozen


    validation_frequency = 5
    global_batch_num = 0
    epoch = 80
    for i in range(epoch):
        print("------ Start epoch {} ------".format(i+1))


        for i_batch, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
 
            left_event = batch['event_volume_left'].cuda()
            right_event = batch['event_volume_right'].cuda()
            left_image = batch['left_image'].cuda()
            right_image = batch['right_image'].cuda()
            disparity_image = batch['disparity_image'].cuda()




            assert model.training
            flow_predictions = model(left_event, right_event, left_image, right_image, iters=args.train_iters)

            assert model.training
            loss, metrics = sequence_loss(flow_predictions,disparity_image,train_loader.dataset)

            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            torch.cuda.empty_cache() 
            total_steps += 1

        save_path = Path('checkpoints/%d_%s.pth' % (i + 1, args.name))
        logging.info(f"Saving file {save_path.absolute()}")
        torch.save(model.state_dict(), save_path)

        if i % validation_frequency == validation_frequency - 1:
                results = validate(model, vali_loader,vali_set,iters=args.valid_iters)

                logger.write_dict(results)

                model.train()
                model.freeze_bn()
       

    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='EGEI-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--batch_size', type=int, default=2, help="batch size used during training.")

    parser.add_argument('--lr', type=float, default=0.0001, help="max learning rate.")
    parser.add_argument('--train_iters', type=int, default=12, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    parser.add_argument('--valid_iters', type=int, default=12, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    # parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")

    parser.add_argument('-p', '--path', type=str, help="Dataset path", required=True)
    parser.add_argument('-s', '--split', default=1, type=int, help="Set of Indoor Flying splits")
    parser.add_argument('-n', '--num_workers', default=1, type=int, help='How many sub-processes to use for data loading')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--mode', default='train', type=str, help="Mode([train/test/demo)")
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)