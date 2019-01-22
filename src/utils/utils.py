import torch
from PIL import Image
from torch.autograd import Variable

import os
import argparse
import time


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize(
            (int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean)
    batch = batch / Variable(std)
    return batch


class Save_text(object):
    """docstring for save_text"""

    def __init__(self):
        super(Save_text, self).__init__()

    def save(self, opt):
        args = vars(opt)
        print((args))
        self.experiment = "checkpoint/experiment_{}".format(opt.num_experiment)
        file_name = os.path.join(
            self.experiment, "opt_{}.txt".format(opt.num_experiment))
        print('-----------------Options-----------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-----------------End-----------------')
        with open(file_name, 'wt') as file:  # closes when the file has modified
            file.write('------------------Setting-----------------\n')
            for option, value in sorted(args.items()):
                file.write('%s: %s\n' % (str(option), str(value)))
            file.write('-------------------End---------------------\n')
            # save current time
            file.write(time.strftime('%Y-%m-%d-%H:%M:%S',
                                     time.localtime(time.time())))
        file.close()
        print("Saving options has done!")
        print(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))

# utils


def check_paths(args):
    save_address = os.path.join(
        args.save_model_dir, "experiment_{}".format(args.num_experiment))
    try:
        if not os.path.exists(args.save_model_dir):
            os.system('mkdir {}'.format(args.save_model_dir))
        if not os.path.exists(save_address):
            os.system('mkdir {}'.format(save_address))
    except OSError as e:
        print(e)
        sys.exit(1)


def get_gpu_list(args):
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    return args
