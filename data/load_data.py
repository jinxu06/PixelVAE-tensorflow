import os
import sys
import numpy as np
from PIL import Image
import data.celeba_data as celeba_data


DataLoader = celeba_data.DataLoader
if args.debug:
    train_data = DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=args.img_size)
else:
    train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=args.img_size)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, size=args.img_size)

class DataSet(object):
    pass

class CelebA(DataSet):

    def __init__(self, data_dir, batch_size, img_size, rng=np.random.RandomState(None)):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.rng = rng

    def train(self, shuffle=True, limit=-1):
        return celeba_data.DataLoader(self.data_dir, 'train', self.batch_size,
                rng=self.rng, shuffle=shuffle, size=self.img_size, limit=limit)

    def test(self, shuffle=False, limit=-1):
        return celeba_data.DataLoader(self.data_dir, 'valid', self.batch_size,
                rng=self.rng, shuffle=shuffle, size=self.img_size, limit=limit)
                

class Cifar10(DataSet):
    pass

class TinyImageNet(DataSet):
    pass
