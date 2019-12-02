# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .duke2market import Duke2Market1501
from .market2duke import Market2Duke
from .dataset_loader import ImageDataset
from .market2cuhk import Market2CUHK

__factory = {
    'market1501': Market1501,
    'duke2market': Duke2Market1501,
    'market2duke': Market2Duke,
    'market2cuhk': Market2CUHK,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
