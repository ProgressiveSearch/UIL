import unittest
import sys
sys.path.append('.')
from config import cfg

from data.datasets import init_dataset


class MyTestCase(unittest.TestCase):
    # def test_data_static(self):
    #     with open('../train.txt', 'r') as f:
    #         train = [i.strip('\n') for i in f.readlines()]
    #     with open('../validation.txt', 'r') as f:
    #         validation = [i.strip('\n') for i in f.readlines()]
    #     with open('../online.txt', 'r') as f:
    #         online = [i.strip('\n') for i in f.readlines()]

    # def test_online_loader(self):
    #     train_loader, val_loader, online_loader, test_loader, num_query = make_online_loader(cfg)
    #     m, n = next(iter(train_loader))
    #     a, b = next(iter(online_loader))
    #     from IPython import embed;
    #     embed()
    def test_duke2market(self):
        dataset = init_dataset('market1501')
        from IPython import embed; embed()


if __name__ == '__main__':
    unittest.main()
