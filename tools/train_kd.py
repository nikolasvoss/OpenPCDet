import _init_path
import argparse
import datetime
import time
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt, eval_single_ckpt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator, load_data_to_gpu
from pcdet.utils import common_utils, commu_utils
from train_utils.optimization import build_optimizer, build_scheduler
# from train_utils.train_utils import train_model
from train_utils.train_utils import save_checkpoint, checkpoint_state, disable_augmentation_hook, train_one_epoch
import tqdm
import gc  # Required for garbage collection

import visual_utils.vis_feature_maps as visfm
import local_paths
from pcdet.utils.spconv_utils import spconv


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str,
                        default=local_paths.cfg_file_multi_train,
                        help='specify the config for training')
    parser.add_argument('--output_dir', type=str, help='specify an output directory if needed',
                        default=None) #local_paths.output_dir_train)
    parser.add_argument('--batch_size', type=int, default=8, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=15, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='can be used to continue training')
    parser.add_argument('--pretrained_model', type=str, help='pretrained_model',
                        default=None)#'/home/niko/Documents/sicherung_trainings/second_s_1_240308/checkpoint_epoch_15.pth')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--eval_after_epoch', action='store_true', default=False, help='evaluate after each epoch')

    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False,
                        help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', default=True, action='store_true', help='use mix precision training')
    parser.add_argument('--v', action='store_true', default=False, help='verbose logging')

    # add arguments for kd loss and entropy loss
    parser.add_argument('--kd_loss_func', type=str, default='entropyRelN',
                        help='kd loss function. Options: basic, entropy, entropyRelN, entropyRelNDense')
    parser.add_argument('--kd_loss_weight', type=float, default=1.0, help='weight for kd loss')
    parser.add_argument('--gt_loss_weight', type=float, default=1.0, help='weight for gt loss')
    parser.add_argument('--num_bins', type=int, default=None, help='number of bins for entropy histogram')
    parser.add_argument('--use_batch_act', action='store_true', default=True,
                        help='use batchnorm + activation for kd loss calculation')
    parser.add_argument('--x_shift', type=float, default=0.5, help='x-shift (threshold) for entropy sigmoid')
    parser.add_argument('--multiplier', type=float, default=15, help='multiplier (edge steepness) for entropy sigmoid')
    parser.add_argument('--lower_bound', type=float, default=0.05,
                        help='lower bound for entropy loss. All values below this are set to 0')
    parser.add_argument('--activation', type=str, default=None, help='activation function used after entropy calculation')
    parser.add_argument('--top_n', type=int, default=5000, help='top n voxels to consider for entropy calculation')
    parser.add_argument('--top_n_relative', type=float, default=0.5, help='top n voxels to consider for entropyRelativeN calculation')

    parser.add_argument('--pretrained_model_teacher', type=str, help='pretrained model for teacher',
                        default=local_paths.pretrained_model_teacher_multi)
    parser.add_argument('--layer0_name_teacher', type=str, default="backbone_2d.blocks.0.16", help='layer0 name for teacher')
    parser.add_argument('--layer1_name_teacher', type=str, default=None, help='layer1 name for teacher')
    parser.add_argument('--layer2_name_teacher', type=str, default=None, help='layer2 name for teacher')
    parser.add_argument('--layer0_name_student', type=str, default="backbone_3d.feat_adapt_single.0", help='layer0 name for student')
    parser.add_argument('--layer1_name_student', type=str, default=None, help='layer1 name for student')
    parser.add_argument('--layer2_name_student', type=str, default=None, help='layer2 name for student')
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def init(args, cfg):
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    if args.output_dir is None:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    else:
        output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    return output_dir, ckpt_dir, logger, tb_log, dist_train, total_gpus

def main():
    torch.backends.cudnn.benchmark = True

    args, cfg = parse_config()

    output_dir, ckpt_dir, logger, tb_log, dist_train, total_gpus = init(args, cfg)

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    ###########################################################################
    # build student model
    ###########################################################################

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer,
                                                           logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))

        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(
        f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    if args.v:  # verbose logging
        logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    ###########################################################################
    # build teacher model
    ###########################################################################
    model_teacher = build_network(model_cfg=cfg.MODEL_TEACHER, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model_teacher.load_params_from_file(filename=args.pretrained_model_teacher, to_cpu=dist_train, logger=logger)
    model_teacher.cuda()
    model_teacher.eval()
    logger.info(
        f'----------- Model Teacher {cfg.MODEL_TEACHER.NAME} created, param count: {sum([m.numel() for m in model_teacher.parameters()])} -----------')
    if args.v:  # verbose logging
        logger.info(model_teacher)

    # Freeze the entire model
    for param in model_teacher.parameters():
        param.requires_grad = False

    # Unfreeze the feat_adapt_autoencoder layer if it exists
    if getattr(model_teacher.backbone_3d, 'feat_adapt_autoencoder', None) is not None:
        for param in model_teacher.backbone_3d.feat_adapt_autoencoder.parameters():
            param.requires_grad = True

    ############################################################################
    # Create hooks for student and teacher model
    createHooks(model, model_teacher, args, logger)

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_kd_model(
        model,
        model_teacher,
        optimizer,
        train_loader,
        eval_after_epoch=args.eval_after_epoch,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
        args=args,
        output_dir=output_dir,
        dist_train=dist_train
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Prepare evaluation during training %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval,
                           0)  # Only evaluate the last args.num_epochs_to_eval epochs

    torch.cuda.empty_cache()
    gc.collect()
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )

    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
def train_kd_model(model, model_teacher, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False, eval_after_epoch=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None,
                show_gpu_stat=False, cfg=None, args=None, output_dir=None, dist_train=None):
    accumulated_iter = start_iter

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None)
    augment_disable_flag = False

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            augment_disable_flag = disable_augmentation_hook(hook_config, dataloader_iter, total_epochs, cur_epoch, cfg,
                                                             augment_disable_flag, logger)
            # with torch.autograd.detect_anomaly():
            # use for debugging
            accumulated_iter = train_one_epoch_kd(
                model, model_teacher, optimizer, train_loader, model_func,
                args=args,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,

                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval,
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            if eval_after_epoch:
                torch.cuda.empty_cache()
                gc.collect()
                # Evaluate the model after each epoch
                eval_epoch(model, cur_epoch, cfg, args, output_dir, logger, dist_train,
                           ckpt_path=ckpt_name.parent / (ckpt_name.name + '.pth')
                           )


def train_one_epoch_kd(model, model_teacher, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None,
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False,
                    use_amp=False, args=None):
    """
    This function is derived from train_utils.train_one_epoch with an added teacher model.
    The teacher is only used for inference and kd loss calculation.
    """
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0 ** 16))

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter, cur_epoch)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model_teacher.eval()
        with torch.cuda.amp.autocast(enabled=use_amp):
            load_data_to_gpu(batch)

            model_teacher(batch)
            # data from hooked layer is stored in visfm.feature_maps

        model.train()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            gt_loss, tb_dict, disp_dict = model_func(model, batch)

        ####################################################
        # KD Loss calculation
        ####################################################
        start_time = time.time()

        if args.kd_loss_func == 'entropy':
            if len(visfm.feature_maps) == 2:
                kd_loss = loss_fmap_entr(visfm.feature_maps[1], visfm.feature_maps[0], args.num_bins, top_n=args.top_n)
            elif len(visfm.feature_maps) == 4:
                kd_loss = loss_fmap_entr(visfm.feature_maps[2], visfm.feature_maps[0], args.num_bins, top_n=args.top_n)
                kd_loss += loss_fmap_entr(visfm.feature_maps[3], visfm.feature_maps[1], args.num_bins, top_n=args.top_n)
            elif len(visfm.feature_maps) == 6:
                kd_loss = loss_fmap_entr(visfm.feature_maps[3], visfm.feature_maps[0], args.num_bins, top_n=args.top_n)
                kd_loss += loss_fmap_entr(visfm.feature_maps[4], visfm.feature_maps[1], args.num_bins, top_n=args.top_n)
                kd_loss += loss_fmap_entr(visfm.feature_maps[5], visfm.feature_maps[2], args.num_bins, top_n=args.top_n)
            else:
                raise ValueError("Invalid number of feature maps. Must be 2, 4 or 6")
        elif args.kd_loss_func == 'entropyRelN':
            kd_loss = loss_fmap_entr_reln_sparse(visfm.feature_maps[1], visfm.feature_maps[0],
                                                 args.num_bins, top_n_relative=args.top_n_relative,
                                                 use_batch_act=args.use_batch_act)
        elif args.kd_loss_func == 'entropyRelNDense':
            kd_loss = loss_fmap_entr_reln_dense(visfm.feature_maps[1], visfm.feature_maps[0],
                                                args.num_bins, top_n_relative=args.top_n_relative,
                                                 use_batch_act=args.use_batch_act)
        elif args.kd_loss_func == 'basic':
            if len(visfm.feature_maps) == 2:
                kd_loss = loss_fmap_kd(visfm.feature_maps[1], visfm.feature_maps[0], use_batch_act=args.use_batch_act)
            elif len(visfm.feature_maps) == 4:
                kd_loss = loss_fmap_kd(visfm.feature_maps[2], visfm.feature_maps[0], use_batch_act=args.use_batch_act)
                kd_loss += loss_fmap_kd(visfm.feature_maps[3], visfm.feature_maps[1], use_batch_act=args.use_batch_act)
            elif len(visfm.feature_maps) == 6:
                kd_loss = loss_fmap_kd(visfm.feature_maps[3], visfm.feature_maps[0], use_batch_act=args.use_batch_act)
                kd_loss += loss_fmap_kd(visfm.feature_maps[4], visfm.feature_maps[1], use_batch_act=args.use_batch_act)
                kd_loss += loss_fmap_kd(visfm.feature_maps[5], visfm.feature_maps[2], use_batch_act=args.use_batch_act)
            else:
                raise ValueError("Invalid number of feature maps. Must be 2, 4 or 6")
        else:
            raise ValueError("Invalid kd_loss_func argument. Must be 'entropy' or 'basic'")

        # Delete the feature maps to prevent errors in backward pass, also frees up memory
        visfm.feature_maps = None

        loss = args.gt_loss_weight * gt_loss + args.kd_loss_weight * kd_loss

        end_time = time.time()  # end time after kd_loss calculation
        kd_loss_time = end_time - start_time  # time taken to calculate kd_loss
        if args.v:  # verbose logging
            logger.info(f"Time for kd_loss calc: {kd_loss_time:.6f}s, "
                  f"weighted GT-Loss: {args.gt_loss_weight*gt_loss:.4f}, "
                  f"weighted KD-Loss: {args.kd_loss_weight*kd_loss:.4f}, "
                  f"total: {loss:.4f}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()

        accumulated_iter += 1

        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)

            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item(), batch_size)

            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * (
                                (total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} '
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch + 1, total_epochs, 100. * (cur_epoch + 1) / total_epochs,
                            cur_it, total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                        )
                    )

                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

            # save intermediate ckpt every {ckpt_save_time_interval} seconds
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter


def eval_epoch(model, epoch, cfg, args, output_dir, logger, dist_train, ckpt_path=None):
    logger.info('**********************Prepare evaluation during training %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # Create eval_output_dir if it does not exist
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the DataLoader for evaluation
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )

    # Conduct the evaluation using the ckpt of the current epoch
    # repeat_eval_ckpt(
    #     model.module if dist_train else model,
    #     test_loader, args, eval_output_dir, logger, ckpt_dir,
    #     dist_test=dist_train
    # )

    # Conduct the evaluation for a single checkpoint
    eval_single_ckpt(
        model, test_loader, args, eval_output_dir, logger, epoch, ckpt_path=ckpt_path
    )

    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # Release memory after evaluation
    del test_loader, test_set, sampler
    gc.collect()


def createHooks(model, model_teacher, args, logger):
    """
    Create hooks for student and teacher model
    They are sorted depending on which model is inferenced first.
    In this case teacher is inferenced first, so the order is:

    if only layer 0 is used:
    feature_map[0] = layer0_teacher
    feature_map[1] = layer0_student

    if all layers are used:
    feature_map[0] = layer0_teacher
    feature_map[1] = layer1_teacher
    feature_map[2] = layer2_teacher
    feature_map[3] = layer0_student
    feature_map[4] = layer1_student
    feature_map[5] = layer2_student
    """
    visfm.registerHookForLayer(model, args.layer0_name_student)
    logger.info('Created student hook for layer: %s' % args.layer0_name_student)
    visfm.registerHookForLayer(model_teacher, args.layer0_name_teacher)
    logger.info('Created teacher hook for layer: %s' % args.layer0_name_teacher)

    if args.layer1_name_student is not None:
        visfm.registerHookForLayer(model, args.layer1_name_student)
        logger.info('Created student hook for layer: %s' % args.layer1_name_student)
    if args.layer1_name_teacher is not None:
        visfm.registerHookForLayer(model_teacher, args.layer1_name_teacher)
        logger.info('Created teacher hook for layer: %s' % args.layer1_name_teacher)
    if args.layer2_name_student is not None:
        visfm.registerHookForLayer(model, args.layer2_name_student)
        logger.info('Created student hook for layer: %s' % args.layer2_name_student)
    if args.layer2_name_teacher is not None:
        visfm.registerHookForLayer(model_teacher, args.layer2_name_teacher)
        logger.info('Created teacher hook for layer: %s' % args.layer2_name_teacher)


def loss_fmap_kd(fmap_student, fmap_teacher, use_batch_act=False):
    """Calculates the KL divergence between the feature maps of the student and teacher networks.
    Firstly the different number of channels must be handled
    """
    if hasattr(fmap_student, 'dense'):
        fmap_student = fmap_student.dense()
        fmap_teacher = fmap_teacher.dense()

    fmap_student = fmap_student.to(torch.float32)
    fmap_teacher = fmap_teacher.to(torch.float32)

    if use_batch_act:
        batch_act = nn.Sequential(nn.BatchNorm3d(fmap_teacher.shape[1], eps=1e-3, momentum=0.01), nn.ReLU()).to(fmap_teacher.device)
        fmap_student = batch_act(fmap_student)
        fmap_teacher = batch_act(fmap_teacher)

    loss = nn.MSELoss()
    return loss(fmap_student, fmap_teacher)


def loss_fmap_entr(fmap_student, fmap_teacher, num_bins=None, top_n=5000):
    """Calculates the entropy of the feature maps of the student network
    1. Sum over the z-axis -> fmap_student.shape [batch, channel, y, x]: [8, 96, 128, 128], fmap_teacher.shape: [8, 128, 128, 128]
    2. Calculate the entropy of the teacher -> entr_teacher.shape: [8, 128, 128]
    3. get top N values of the entropy map -> indices.shape: [8, N, 2] 2: x, y
    4. get top N values of the feature maps of the student and teacher -> topN_student_values.shape: [8, channel, N]
    5. calculate the loss

    """
    if hasattr(fmap_student, 'dense'):
        fmap_student = fmap_student.dense()
        fmap_teacher = fmap_teacher.dense()
    # sum over z axis
    fmap_student = fmap_student.sum(axis=-3, keepdims=False)
    fmap_teacher = fmap_teacher.sum(axis=-3, keepdims=False)
    # check if the shape of the last two dimensions of fmap_student and fmap_teacher are the same
    if fmap_student.shape[-2:] != fmap_teacher.shape[-2:]:
        raise ValueError('Feature maps of student and teacher do not have the same shape in the last two dimensions.')
    entr_teacher, _ = visfm.calc_fmap_entropy_torch(fmap_teacher, num_bins)

    # find top N values for each batch element entr_teacher[batch, :,:]
    # entr_teacher needs to be flattened in order to work with topk
    # indices_flattened.shape = [batch, N]
    # [1] means to only use the second return value of topk
    indices_flattened = torch.topk(torch.flatten(entr_teacher, start_dim=1), top_n, dim=1)[1]

    # convert the flattened indices back to 2D indices
    indices = torch.stack((indices_flattened // entr_teacher.shape[-1], indices_flattened % entr_teacher.shape[-1]), dim=-1)
    # Get the indices for all batches at once
    batch_indices = torch.arange(indices.shape[0]).view(-1, 1, 1).expand_as(indices).to(indices.device)

    # Use advanced indexing to get all the required values at once
    topN_teacher_values = fmap_teacher[batch_indices[:, :, 0], :, indices[:, :, 0], indices[:, :, 1]]
    topN_student_values = fmap_student[batch_indices[:, :, 0], :, indices[:, :, 0], indices[:, :, 1]]

    loss = nn.MSELoss()
    return loss(topN_student_values, topN_teacher_values)

def loss_fmap_entr_reln_dense(fmap_student, fmap_teacher, num_bins=None, top_n_relative=0.5, use_batch_act=False):
    """
    Calculates the Mean Squared Error (MSE) loss between the top N values of the student and teacher feature maps.
    The top N values are determined based on the entropy of the teacher feature map.

    Args:
        fmap_student (torch.Tensor): The feature map of the student network.
        fmap_teacher (torch.Tensor): The feature map of the teacher network.
        num_bins (int, optional): The number of bins used for the entropy calculation. Defaults to None.
        top_n_relative (float, optional): The percentage of top N values to consider for the entropy calculation (0-1). Defaults to 0.75.

    Returns:
        torch.Tensor: The MSE loss between the top N values of the student and teacher feature maps.
    """
    # apply batch norm and ReLU
    if use_batch_act:
        batch_act = nn.Sequential(nn.BatchNorm1d(fmap_teacher.shape[1], eps=1e-3, momentum=0.01), nn.ReLU()).to(fmap_teacher.device)
        fmap_student = batch_act(fmap_student)
        fmap_teacher = batch_act(fmap_teacher)

    # Calculate entropy of the teacher in dense format
    entr_teacher, _ = visfm.calc_fmap_entropy_torch(fmap_teacher, num_bins)

    # Calculate the number of top values to consider
    top_n = int(top_n_relative * entr_teacher[0].numel())
    topN_indices = torch.zeros([entr_teacher.shape[0], top_n], dtype=torch.long, device=entr_teacher.device)
    # Iterate over each batch
    for i in range(fmap_teacher.shape[0]):
        # Get the topN indices of the entropy map
        topN_indices[i] = torch.topk(entr_teacher[i].view(-1), top_n).indices

    # Reshapes the teacher feature map tensor into a 2D tensor and then gathers the top N values
    # from the reshaped tensor using the indices provided by 'topN_indices'. The 'unsqueeze' and 'expand'
    # operations are used to match the dimensions of 'topN_indices' with the teacher feature map tensor for correct indexing.
    topN_teacher_values = (fmap_teacher.view(fmap_teacher.shape[0], fmap_teacher.shape[1], -1)
                           .gather(-1, topN_indices.unsqueeze(1).expand(-1, fmap_teacher.shape[1], -1)))
    topN_student_values = (fmap_student.view(fmap_student.shape[0], fmap_student.shape[1], -1)
                            .gather(-1, topN_indices.unsqueeze(1).expand(-1, fmap_student.shape[1], -1)))

    loss = nn.MSELoss()(topN_student_values, topN_teacher_values)

    return loss


def loss_fmap_entr_reln_sparse(fmap_student, fmap_teacher, num_bins=None, top_n_relative=0.5, use_batch_act=False):
    """Calculates the entropy loss of the with a relative topN value in sparse format
    1. Counts values per batch and calculates the relative topN value
    2. Calculate entropy of the teacher in sparse format
    3. Get topN indices of the entropy map
    4. Get topN values of the feature maps of the student and teacher
    5. Calculate MSE-Loss between the topN values of the student and teacher

    Input:
    - fmap_student: SparseConvTensor of the student network
    - fmap_teacher: SparseConvTensor of the teacher network
    - num_bins (None): number of bins used for the entropy calculation
    - top_n_relative (0.75): percentage of topN values to consider for the entropy calculation (0-1)

    Output:
    - loss: MSE-Loss between the topN values of the student and teacher
    """
    if use_batch_act:
        batch_act = spconv.SparseSequential(nn.BatchNorm1d(fmap_teacher.features.shape[1], eps=1e-3, momentum=0.01),
                                            nn.ReLU()).to(fmap_teacher.features.device)
        fmap_student = batch_act(fmap_student)
        fmap_teacher = batch_act(fmap_teacher)

    batch_counts = torch.bincount(fmap_teacher.indices[:, 0])
    batch_counts_relative = (batch_counts * top_n_relative).int()
    batch_counts = torch.cat((torch.tensor([0], device=batch_counts.device),
                              torch.cumsum(batch_counts, dim=0)))

    entr_teacher = visfm.calc_fmap_entropy_sparse(feature_map=fmap_teacher.features, num_bins=num_bins)

    topN_features_teacher = torch.empty(0, device=fmap_teacher.features.device)
    topN_features_student = torch.empty(0, device=fmap_student.features.device)

    # get topN indices for each batch. The indices must match the concatenated structure of the sparse tensor.
    # Then get the topN values of the feature maps of the student and teacher.
    for batch in range(len(batch_counts)-1):
        topN_indices = torch.topk(entr_teacher[batch_counts[batch]:batch_counts[batch+1]], batch_counts_relative[batch], largest=True).indices
        topN_indices += batch_counts[batch]

        # Concatenate new values to the tensors
        topN_features_teacher = torch.cat((topN_features_teacher, fmap_teacher.features[topN_indices]), dim=0)
        topN_features_student = torch.cat((topN_features_student, fmap_student.features[topN_indices]), dim=0)

    loss = nn.MSELoss()
    return loss(topN_features_student, topN_features_teacher)


if __name__ == '__main__':
    main()