import os
import tqdm
import pickle
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from . import consts


def listdir(path):
    return [d for d in os.listdir(path) if not d.startswith('.')]

class WashHandDataset(torch.utils.data.Dataset):
    def __init__(self, root, cache=None, transform=None):
        self.transform = transform if transform is not None else lambda x: x
        self.c_list = listdir(root)
        self.images = []
        self.labels = []
        for i, c in enumerate(self.c_list):
            img_path_list = listdir(os.path.join(root, c))
            for f in img_path_list:
                f = os.path.join(root, c, f)
                # with Image.open(f) as fp:
                #     self.images.append(fp)
                self.images.append(f)
                self.labels.append(consts.l2n[c])

    def __getitem__(self, idx):
        im = np.array(Image.open(self.images[idx]))
        return self.transform(im), self.labels[idx]

    def __len__(self):
        return len(self.images)
