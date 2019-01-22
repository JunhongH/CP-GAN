import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import random
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class TrainDataset(data.Dataset):
    def __init__(self, dataset, style_dataset, transform=None, style_transform=None, is_train=1, split_rate=0.15):
        #self.opt = args
        self.dir_A = dataset
        self.dir_B = style_dataset
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = transform
        self.style_transform = style_transform
        self.img_id_lis = list(range(0, len(self.A_paths)))
        self.img_id_lis = np.array(self.img_id_lis)
        self.is_train = is_train
        self.split_rate = split_rate
        random.seed(1)
        index_lis = list(range(0, self.img_id_lis.shape[0]))
        random.shuffle(index_lis)
        # print(index_lis[0:10])
        if self.is_train:
            index_lis = index_lis[int(len(index_lis) * self.split_rate):]
        else:
            index_lis = index_lis[0:int(len(index_lis) * self.split_rate)]

        # get sample for train or eval
        self.img_id_lis = self.img_id_lis[index_lis]
        self.A_size = len(self.img_id_lis)
        print("the length of dataset is {}".format(self.A_size))

    def __getitem__(self, index):
        A_path = self.A_paths[self.img_id_lis[index % self.A_size]]
        B_path = self.B_paths[index % self.B_size]
        # print(A_path)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.style_transform(B_img)

        return [A_img, B_img]

    def __len__(self):
        # return max(self.A_size, self.B_size)
        return self.A_size

    def name(self):
        return 'TrainDataset'


class TestDataset(data.Dataset):
    def __init__(self, test_dataset, transform=None):
        #self.opt = args
        self.dir_A = test_dataset
        # self.dir_B = style_dataset
        self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        # print(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        # self.B_size = len(self.B_paths)
        self.transform = transform
        # self.style_transform=style_transform

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.B_size]
        # print(A_path)
        A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        # B_img = self.style_transform(B_img)

        return A_img

    def __len__(self):
        return self.A_size

    def name(self):
        return 'TestDataset'
