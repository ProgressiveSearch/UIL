# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse, os, sys, time
import numpy as np
from collections import defaultdict
import torch
from torch.backends import cudnn
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_online_train
from engine.inference import inference
from modeling import build_model
from modeling.baseline import End2End_AvgPooling
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger


def online_train(cfg):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    online_model = End2End_AvgPooling(1, 0.5, 2048, 0)
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
    #                                     '/duke2market/resnet50_model_350.pth'))
    online_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
                                        '/market2duke/resnet50_model_350.pth'))

    online_model.to('cuda')

    arguments = {}

    loss_func = make_loss(cfg)

    # prepare online dataset
    online_train = [list(d) for d in online_train]
    if cfg.DATASETS.ONLINE == 'market1501':
        all_ids = list(np.arange(0, 751))
    else:
        all_ids = list(np.arange(0, 703))
    increment_id = 100
    for i in range(7):
        chosed_id = np.random.choice(all_ids, size=increment_id, replace=False)
        online_set = list(filter(lambda x: x[1] in chosed_id, online_train))

        # reorder index
        for k, d in enumerate(online_set):
            d[3] = k
    # for d in online_train:
    #     if d[1] < current_id:
    #         online_dict[chosed_id].append(list(d))
    #     else:
    #         current_id += increment_id
    #         chosed_id += 1

    # ==========
    # for on_step in online_dict:
    #     online_set = online_dict[on_step]
    #     # reorganize index
    #     for i, d in enumerate(online_set):
    #         d[3] = i

        cluster_model = End2End_AvgPooling(1, 0.5, 2048, 0)
        # cluster_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
        #                                     '/duke2market/resnet50_model_350.pth'))
        cluster_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
                                            '/market2duke/resnet50_model_350.pth'))
        # cluster_model.load_weight(online_model.state_dict())

        state_dict = do_online_train(i, cfg, cluster_model, online_set, val_loader, num_query, experiment)

        torch.save(state_dict, cfg.OUTPUT_DIR + '/duke_model_{}.pth'.format(i))

        # update online model
        for key, value in state_dict.items():
            if key in online_model.state_dict():
                online_param = online_model.state_dict()[key].data
                online_model.state_dict()[key].data.copy_(0.9 * online_param + 0.1 * value.data)

        # online_model.load_weight(state_dict)
        # test online model performance
        inference(cfg, online_model, val_loader, num_query, experiment)
    # ==========


def online_test(cfg, logger):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    # online_model = End2End_AvgPooling(1, 0.5, 2048, 0)
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
    #                                     '/duke2market/resnet50_model_350.pth'))
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
    #                                     '/market2duke/resnet50_model_350.pth'))


    # online_model.to('cuda')

    for alpha in np.arange(0, 1.1, 0.1):
        online_model = End2End_AvgPooling(1, 0.5, 2048, 0)
        online_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
                                        '/duke2market/resnet50_model_350.pth'))
        online_model.to('cuda')
        for on_step in range(7):
            state_dict = torch.load('/export/home/lxy/online-reid/iccv_logs'
                                    '/online.04_02_09:37:32/market_model_{}.pth'.format(on_step))
            # state_dict = torch.load('/export/home/lxy/online-reid/iccv_logs'
            #                         '/online.04_02_09:38:12/duke_model_{}.pth'.format(on_step))
            if on_step == 6:
                alpha /= 2
            # update online model
            for key, value in state_dict.items():
                if key in online_model.state_dict():
                    online_param = online_model.state_dict()[key].data
                    online_model.state_dict()[key].data.copy_((1-alpha) * online_param + alpha * value.data)

            # test online model performance
        logger.info('alpha = {:.1f}'.format(alpha))
        inference(cfg, online_model, val_loader, num_query, 0)
    # ==========

def cross_train(cfg):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    online_model = End2End_AvgPooling(1, 0.5, 2048, 0)
    online_model.load_weight(torch.load('./iccv_logs/market2cuhk.10_21_00:33:01/resnet50_model_350.pth'))

    online_model.to('cuda')
    # inference(cfg, online_model, val_loader, num_query, experiment)


    arguments = {}

    loss_func = make_loss(cfg)

    # cluster merge
    do_online_train(0, cfg, online_model, online_train, val_loader, num_query)

def train(cfg):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = End2End_AvgPooling(1, 0.5, 2048, num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, 
                                  cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_loss(cfg)

    arguments = {}

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query,
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    output_dir = os.path.join(os.getcwd() + '/iccv_logs', cfg.OUTPUT_DIR + time.strftime(".%m_%d_%H:%M:%S"))

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.CUDA)

    logger = setup_logger("reid_online", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True


    # online_train(cfg)
    # train(cfg)
    cross_train(cfg)
    # online_test(cfg, logger)


if __name__ == '__main__':
    main()
