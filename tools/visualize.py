import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from tensorboardX import SummaryWriter

# from visual_utils.vis_feature_maps import register_hook_for_layer, visualize_feature_map
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/nvoss/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_second_S_multihead.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='/home/nvoss/OpenPCDet/output/home/nvoss/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_second_S_multihead/default/ckpt/checkpoint_epoch_15.pth', help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, help='pretrained_model', default=None)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def extract_feature_map_hook(module, input, output):
    """Extracts the output of the layer for visualization."""
    global feature_maps
    feature_maps = output

def register_hook_for_layer(model, layer_name):
    """Registers a hook for the specified layer in the model."""
    for name, module in model.named_modules():
        if name == layer_name:
            # Register a forward hook on the layer to capture feature maps
            module.register_forward_hook(extract_feature_map_hook)

def visualize_feature_map(feature_map, output_dir, unknown_idx, fmap_idx=None, z_plane=None):
    """Visualizes a slice of one feature map using Matplotlib."""
    visibility_factor = 4 # multiply all values for better visibility

    if feature_map is not None:
        feature_map = feature_map.dense()
        plt.figure(figsize=(8, 8)) # Set the figure size to be 8x8 inches
        # Detach the feature map from GPU and convert to NumPy
        for fmap_idx in range(len(feature_map[unknown_idx])):
            for z_plane in range(len(feature_map[unknown_idx][fmap_idx])):
                feature_slice = feature_map[unknown_idx][fmap_idx][z_plane].cpu().detach().numpy()
                plt.imshow(feature_slice*visibility_factor, cmap='PRGn', vmax=1, vmin=-1)
                plt.colorbar()
                file_name = os.path.join(output_dir, (f'map_{unknown_idx}_fmap{fmap_idx}_z{z_plane}.jpg'))
                plt.savefig(os.path.join(output_dir, file_name), dpi=150) # > 1024x1024 pixels (8 inches * 128 DPI)
                plt.clf()
    else:
        print("No feature map available. Check if the hook was triggered correctly.")

def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    # set correct batch size for each gpu
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    # eval_output_dir = Path('~/OpenPCDet/tools')
    # log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    log_file = 'visualizer.log'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # build model and load weights
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test,
                                    pre_trained_path=args.pretrained_model)
    model.cuda()
    model.eval()

    layer_name = "backbone_3d.conv1.0.conv1"  # Replace with the layer you want to visualize
    unknown_idx = 0  # TODO: what is this?
    fmap_idx = 4  # Index of the feature map to visualize
    z_plane_idx = None  # Index of the z-plane to visualize from the 3D feature map
    output_dir = f'/home/nvoss/OpenPCDet/feature_map_saves/{layer_name}/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    register_hook_for_layer(model, layer_name)

    input_dict = next(iter(test_loader))
    

    with torch.no_grad():
        # Forward pass through the model
        load_data_to_gpu(input_dict)
        
        pred_dicts, ret_dict = model.forward(input_dict)

    # Visualize the feature map
    visualize_feature_map(feature_maps, output_dir, unknown_idx, fmap_idx, z_plane_idx)


if __name__ == '__main__':
    main()
