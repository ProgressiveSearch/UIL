# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import copy
import logging

import numpy as np
import torch.nn.functional as F
import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from data.build import get_dataloader
from layers.exclusive_loss import ExLoss
from solver import make_optimizer, WarmupMultiStepLR
from utils.reid_metric import R1_mAP


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.cuda()
        target = target.cuda()
        rep_feat, feat, score = model(img)
        loss = loss_fn(score, feat, target)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            feat = model(data)[1]
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_online_trainer(model, optimizer, u_criterion, alpha):

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        if u_criterion.C.grad is not None:
            u_criterion.C.grad.detach_()
            u_criterion.C.grad.zero_()

        # unlabel data
        u_img, _, _, indexs = batch
        target = indexs.cuda()

        u_img = u_img.cuda()
        feats, eval_feat = model(u_img)

        # repelled loss
        rep_loss, outputs = u_criterion(feats, target)

        if alpha == 0:
            loss = rep_loss
            loss.backward()
            optimizer.step()
        else:
            # appealed loss
            cluster_center = u_criterion.C[indexs]
            app_loss = 0.5 * ((feats - cluster_center) ** 2).sum(dim=1).mean()
            loss = rep_loss + alpha * app_loss
            loss.backward()
            optimizer.step()
            u_criterion.C.data.add_(-0.1 * u_criterion.C.grad.data)

        u_acc = (outputs.max(1)[1] == target).float().mean()

        return loss.item(), u_acc.item()

    return Engine(_update)


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_online.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10, 20]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    trainer.run(train_loader, max_epochs=epochs)


def do_online_train(
        on_step,
        cfg,
        model,
        online_train,
        val_loader,
        num_query,
):
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE

    cluster_id_labels = [i[3] for i in online_train]
    num_train_ids = len(cluster_id_labels)
    nums_to_merge = int(num_train_ids * 0.05)

    logger = logging.getLogger("reid_online.train")
    logger.info("Online dataset {} start training".format(on_step))

    model.to(device)

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query)}, device=device)

    u_dataloader = get_dataloader(cfg, online_train, is_train=False)
    u_label = np.array([label for _, label, _, _ in online_train])

    def calculate_distance(u_feas):
        # calculate distance between features
        x = torch.from_numpy(u_feas)
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists

    def select_merge_data(u_feas, label, label_to_images, ratio_n, dists):
        dists.add_(torch.tril(100000 * torch.ones(len(u_feas), len(u_feas))))

        # 每个聚类中心图片的张数
        cnt = torch.FloatTensor([len(label_to_images[label[idx]]) for idx in range(len(u_feas))])
        dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))  # size penalty, 保持每个聚类中心数量一致

        for idx in range(len(u_feas)):  # 遍历每一张图片
            for j in range(idx + 1, len(u_feas)):  # 只考虑右上角的元素
                if label[idx] == label[j]:  # 如果属于这个聚类中心，把他的距离置为100000
                    dists[idx, j] = 100000

        dists = dists.numpy()
        ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        idx1 = ind[0]
        idx2 = ind[1]
        return idx1, idx2

    def generate_new_train_data(idx1, idx2, label, num_to_merge):
        correct = 0
        num_before_merge = len(np.unique(np.array(label)))  # number of cluster before merge
        # merge clusters with minimum dissimilarity
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1 < label2:  # merge minimum distance
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]  # move to smaller label
            if u_label[idx1[i]] == u_label[idx2[i]] and label1 != label2:  # if label match, correct count add 1
                correct += 1
            num_merged = num_before_merge - len(np.sort(np.unique(np.array(label))))
            if num_merged == num_to_merge:  # if reach merged number, break
                break
        # set new label to the new training data
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]  # rearrange label to 0 - n
        new_train_data = []
        for idx, data in enumerate(online_train):
            new_data = list(copy.deepcopy(data))
            new_data[3] = label[idx]
            new_train_data.append(new_data)

        num_after_merge = len(np.unique(np.array(label)))  # remaining cluster center
        logger.info("num of label before merge: {}, after_merge: {}, sub: {}, Prec: {}/{}={:.3f}"
                    .format(num_before_merge, num_after_merge, num_before_merge - num_after_merge,
                            correct, num_before_merge - num_after_merge, correct/(num_before_merge - num_after_merge)))

        return new_train_data, label

    def get_new_train_data(labels, nums_to_merge, size_penalty):
        # generate average feature
        model.eval()
        fcs = []
        pool5s = []
        for imgs, label, camid, index in u_dataloader:
            with torch.no_grad():
                _fc, pool5 = model(imgs.cuda())
                fcs.append(_fc.cpu())
                pool5s.append(pool5.cpu())
        fcs = torch.cat(fcs, dim=0).numpy()
        u_feas = torch.cat(pool5s, dim=0)
        eval_feas = F.normalize(u_feas).numpy()
        u_feas = u_feas.numpy()

        label_to_images = {}
        for idx, l in enumerate(labels):
            label_to_images[l] = label_to_images.get(l, []) + [idx]

        # calculate average feature/classifier of a cluster
        feature_avg = np.zeros((len(label_to_images), len(u_feas[0])))  # [cluster, feat_dim]
        fc_avg = np.zeros((len(label_to_images), len(fcs[0])))
        for l in label_to_images:
            feature_avg[l] = np.mean(u_feas[label_to_images[l]], axis=0)
            fc_avg[l] = np.mean(fcs[label_to_images[l]], axis=0)

        # calculate distance between features
        dists = calculate_distance(eval_feas)

        idx1, idx2 = select_merge_data(u_feas, labels, label_to_images, size_penalty, dists)

        new_train_data, labels = generate_new_train_data(idx1, idx2, labels, nums_to_merge)

        num_train_ids = len(np.unique(np.array(labels)))

        # change the criterion classifer
        criterion = ExLoss(2048, fc_avg.shape[0], t=10).cuda()
        new_classifier = fc_avg.astype(np.float32)
        new_cluster_center = feature_avg.astype(np.float32)
        criterion.V.data.copy_(F.normalize(torch.from_numpy(new_classifier)).cuda())
        criterion.C.data.copy_(F.normalize(torch.from_numpy(new_cluster_center)).cuda())

        return labels, new_train_data, criterion

    new_train_data = online_train
    labels = [d[3] for d in new_train_data]
    criterion = ExLoss(2048, len(online_train), t=10).cuda()
    total_step = int(1 / 0.05) - 1
    for step in range(total_step):
        logger.info('step: {}/{}'.format(step + 1, int(1 / 0.05) - 1))

        if step == 0:
            epochs = 20
            alpha = 0
        elif step < 10:
            epochs = 3
            alpha = 0.1
        elif step < 15:
            epochs = 2
            alpha = 0.1
        else:
            epochs = 1
            alpha = 0

        epochs = 20 if step == 0 else 2
        # prepare dataset
        online_loader = get_dataloader(cfg, new_train_data, is_train=True)
        # prepare optimizer
        optimizer = make_optimizer(cfg, model, step)
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        # prepare trainer
        trainer = create_online_trainer(model, optimizer, criterion, alpha)
        # checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
        timer = Timer(average=True)

        # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
        #                                                                  'optimizer': optimizer.state_dict()})
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # average metric to attach on trainer
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_u_acc')

        @trainer.on(Events.EPOCH_STARTED)
        def adjust_learning_rate(engine):
            scheduler.step()

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(online_loader) + 1
            log_period = len(online_loader) // 2
            if iter % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Unlabel Acc: {:.3f}, "
                            "Base Lr: {:.2e}"
                            .format(engine.state.epoch, iter, len(online_loader),
                                    engine.state.metrics['avg_loss'], engine.state.metrics['avg_u_acc'],
                                    scheduler.get_lr()[0]))

        # adding handlers using `trainer.on` decorator API
        @trainer.on(Events.EPOCH_COMPLETED)
        def print_times(engine):
            logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                        .format(engine.state.epoch, timer.value() * timer.step_count,
                                online_loader.batch_size / timer.value()))
            logger.info('-' * 10)
            timer.reset()

        # train
        trainer.run(online_loader, max_epochs=epochs)

        # evaluation
        evaluator.run(val_loader)
        cmc, mAP = evaluator.state.metrics['r1_mAP']
        logger.info("Validation Results - Step: {}".format(step))
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10, 20]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        #
        # with experiment.test():
        #     experiment.log_metric('online_mAP', mAP)
        #     experiment.log_metric('online_rank1', cmc[0])

        logger.info('------bottom-up cluster-------')
        # get new cluster id and datasets
        labels, new_train_data, criterion = get_new_train_data(labels, nums_to_merge, 0.005)
    return model.state_dict()
