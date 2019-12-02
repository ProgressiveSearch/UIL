# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, test_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def get_dataloader(cfg, train_data, is_train=True):
    transform = build_transforms(cfg, is_train=is_train)
    batch_size = cfg.SOLVER.IMS_PER_BATCH if is_train else cfg.TEST.IMS_PER_BATCH
    data_loader = DataLoader(
        ImageDataset(train_data, transform=transform),
        batch_size=batch_size, num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=is_train, pin_memory=True, drop_last=is_train
    )
    return data_loader


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    test_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.DATASETS.BASE)

    # train set
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    # online set
    onlineset = init_dataset(cfg.DATASETS.ONLINE)
    online_train = onlineset.train

    # query/gallery set
    test_set = ImageDataset(onlineset.query + onlineset.gallery, test_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn
    )
    return train_loader, online_train,  test_loader, len(onlineset.query), num_classes


def make_online_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    test_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        with open(cfg.DATASETS.TRAIN, 'r') as f:
            train = [i.strip('\n') for i in f.readlines()]
        with open(cfg.DATASETS.VALIDATION, 'r') as f:
            validation = [i.strip('\n') for i in f.readlines()]
        with open(cfg.DATASETS.ONLINE, 'r') as f:
            online = [i.strip('\n') for i in f.readlines()]
        dataset = init_dataset(cfg.DATASETS.NAMES[0], trainset=train, validationset=validation, onlineset=online)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES)

    # train set
    train_set = ImageDataset(dataset.train, train_transforms)
    train_loader = DataLoader(
        train_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    # validation set
    validation_set = ImageDataset(dataset.validation, test_transforms)
    valid_loader = DataLoader(
        validation_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn
    )

    # online set
    online_set = ImageDataset(dataset.online_train, train_transforms)
    online_loader = DataLoader(
        online_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )

    # query/gallery set
    test_set = ImageDataset(dataset.query + dataset.gallery, test_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=test_collate_fn
    )
    return train_loader, valid_loader, online_loader, test_loader, len(dataset.query)
