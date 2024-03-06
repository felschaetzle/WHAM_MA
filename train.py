import os
import os.path as osp
from time import time

import torch
import pprint
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from configs.config import parse_args
from lib.core.trainer import Trainer
from lib.core.loss import WHAMLoss

from lib.utils.utils import prepare_output_dir
from lib.data.dataloader import setup_dloaders
from lib.utils.utils import create_logger, get_optimizer
from lib.models import build_network, build_body_model

def setup_seed(seed):
    """ Setup seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(cfg):
    
    logger = create_logger(cfg.LOGDIR, phase='debug' if cfg.DEBUG else 'train')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
    logger.info(pprint.pformat(cfg))
    
    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)
    
    # ========= Dataloaders ========= #
    data_loaders = setup_dloaders(cfg, cfg.TRAIN.DATASET_EVAL, 'test')
    logger.info(f'Dataset loaded')
    
    # ========= Network and Optimizer ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    optimizer = get_optimizer(
        cfg,
        model=network, 
        optim_type=cfg.TRAIN.OPTIM,
        momentum=cfg.TRAIN.MOMENTUM,
        stage=cfg.TRAIN.STAGE)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 70],
        gamma=0.1,
        verbose=True,
    )
    
    # ========= Loss function ========= #
    criterion = WHAMLoss(cfg, cfg.DEVICE)
    
    # Seed
    if cfg.SEED_VALUE >= 0:
        setup_seed(cfg.SEED_VALUE)

    # ========= Start Training ========= #
    Trainer(
        data_loaders=data_loaders,
        network=network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        train_stage=cfg.TRAIN.STAGE,
        start_epoch=cfg.TRAIN.START_EPOCH,
        end_epoch=cfg.TRAIN.END_EPOCH,
        checkpoint=cfg.TRAIN.CHECKPOINT,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        resume=cfg.RESUME,
        logdir=cfg.LOGDIR,
        summary_iter=cfg.SUMMARY_ITER,
    ).fit()
    
    
if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg)