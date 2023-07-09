import numpy as np
from torch.utils.data import Dataset

from .io import read_img


class ImgDataset(Dataset):
    def __init__(self, img_paths, bitpack=False, bitpack_dim=None, **reader_kwargs):
        self.paths = img_paths
        self.bitpack = bitpack
        self.bitpack_dim = bitpack_dim
        self.reader_kwargs = reader_kwargs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        im, _ = read_img(self.paths[item], **self.reader_kwargs)

        if self.bitpack:
            im = im >= 0.5
            im = np.packbits(im, axis=self.bitpack_dim - 1)
        else:
            im = (im * 255).astype(np.uint8)
        return item, im
