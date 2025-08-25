import argparse
import datetime
import glob
import os
import shutil  # 新增：用于跨平台文件复制
from pathlib import Path

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from ptt.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from ptt.datasets import build_dataloader
from ptt.models import build_network, model_fn_decorator
from ptt.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/ptt.yaml', help='config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='trailer80', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, help='pretrained_model',
                        default=None)
    # 移除分布式相关参数（保留但不生效，避免命令行传入时报错）
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='仅支持none')
    parser.add_argument('--tcp_port', type=int, default=16666, help='分布式训练端口（已禁用）')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='是否使用同步BN（单卡无效）')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='保存checkpoint的间隔epoch')
    parser.add_argument('--local_rank', type=int, default=0, help='本地rank（单卡固定为0）')
    parser.add_argument('--max_ckpt_save_num', type=int, default=80, help='最大保存checkpoint数量')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='设置额外的配置参数')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # 移除'cfgs'和'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    # 强制指定仅使用0号显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args, cfg = parse_config()

    # 禁用分布式训练，强制单卡模式
    dist_train = False
    total_gpus = 1
    cfg.LOCAL_RANK = 0  # 固定本地rank为0

    # 处理batch size（单卡模式下直接使用配置的批次大小）
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        # 单卡模式下无需除以GPU数量
        args.batch_size = args.batch_size

    # 处理训练epochs
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    # 固定随机种子
    if args.fix_random_seed:
        import random
        random.seed(1)
        torch.manual_seed(1)
        worker_init = None
    else:
        worker_init = None

    # 创建输出目录
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 初始化日志
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 日志输出系统信息
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # 输出参数信息
    logger.info('total_batch_size: %d' % args.batch_size)  # 单卡模式下直接输出batch size
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # 复制配置文件（使用shutil跨平台复制）
    if cfg.LOCAL_RANK == 0:
        shutil.copy(args.cfg_file, str(output_dir))  # 替换os.system('cp ...')

    # 初始化tensorboard
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------创建数据加载器、网络和优化器---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,  # 传入False禁用分布式采样
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        worker_init_fn=worker_init
    )
    if cfg.TRAIN.WITH_EVAL.ENABLE:
        test_set, test_loader, test_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train,  # 传入False禁用分布式采样
            workers=args.workers,
            logger=logger,
            training=False,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            total_epochs=args.epochs,
            collate=lambda x: x,
            worker_init_fn=worker_init
        )
    else:
        test_set = test_loader = test_sampler = None

    # 构建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # 单卡模式下禁用sync_bn（同步BN仅在多卡有效）
    if args.sync_bn:
        logger.warning("单卡模式下不支持SyncBatchNorm，已自动禁用")
    model.cuda()  # 直接将模型放到0号显卡

    # 构建优化器
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 加载 checkpoint 或预训练模型
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=False, logger=logger)  # 单卡无需转CPU

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(
            args.ckpt, to_cpu=False, optimizer=optimizer, logger=logger  # 单卡无需转CPU
        )
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=False, optimizer=optimizer, logger=logger  # 单卡无需转CPU
            )
            last_epoch = start_epoch + 1

    model.train()  # 训练模式

    # 单卡模式下不使用DistributedDataParallel
    logger.info(model)

    # 构建学习率调度器
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------开始训练---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(
        cfg=cfg,
        logger=logger,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
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
        loss_record_handle=common_utils.AverageMeter
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    # Windows系统多进程支持（可选）
    import platform
    if platform.system() == "Windows":
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    main()