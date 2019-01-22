import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_net import TransformerNet
import models.networks as networks


class GAN():
    def __init__(self, args):
        self.args = args
        Tensor = torch.cuda.FloatTensor if args.gpu_ids else torch.Tensor
        use_sigmoid = args.no_lsgan

        # Global discriminator

        self.netD = networks.define_D(
            args.input_nc, args.ndf, args.which_model_netD, args.n_layers_D, args.norm, use_sigmoid, args.gpu_ids)

        # Local discriminator

        self.netD_local = networks.define_D(
            args.input_nc, args.ndf, args.which_model_netD, args.n_layers_D, args.norm, use_sigmoid, args.gpu_ids)

        # Generator

        self.netG = TransformerNet(args.norm, args.affine_state)

        self.gan_loss = networks.GANLoss(
            use_lsgan=not args.no_lsgan, tensor=Tensor)

        self.identity_criterion = torch.nn.L1Loss()

        # Resume

        if args.resume_netG != '':
            self.netG.load_state_dict(torch.load(args.resume_netG))
        if args.resume_netD != '':
            self.netD.load_state_dict(torch.load(args.resume_netD))
        if args.resume_netD_local != '':
            self.netD_local.load_state_dict(torch.load(args.resume_netD_local))

        # optimizer

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizer_D_local = torch.optim.Adam(
            self.netD_local.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        if self.args.cuda:
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()

    def update_D(self, content, fake):
        self.netD.volatile = False
        # feed real image
        pred_real = self.netD(content)
        loss_D_real = self.gan_loss(pred_real, True)
        # feed fake image
        pred_fake = self.netD(fake.detach())
        loss_D_fake = self.gan_loss(pred_fake, False)
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5 * self.args.gan_weight
        return self.loss_D

    def update_G(self, fake):
        self.netD.volatile = True
        pred_fake = self.netD(fake)
        self.loss_G = self.gan_loss(pred_fake, True) * self.args.gan_weight
        return self.loss_G

    def update_D_local(self, content_patch, fake_patch):
        self.netD_local.volatile = False

        # feed real sample

        pred_real = self.netD_local(content_patch)
        local_D_real = self.gan_loss(pred_real, True)

        # feed fake sample

        pred_fake = self.netD_local(fake_patch.detach())
        local_D_fake = self.gan_loss(pred_fake, False)
        self.loss_D_local = (local_D_real + local_D_fake) * \
            0.5 * self.args.gan_local_weight
        return self.loss_D_local

    def update_G_local(self, fake_patch):
        self.netD_local.volatile = True
        pred_fake = self.netD_local(fake_patch)
        self.loss_G_local = self.gan_loss(
            pred_fake, True) * self.args.gan_local_weight
        return self.loss_G_local

        # the cropping and padding scheme

    def crop_and_pad(self, image):

        # cropping

        image_crop_1 = image[:, :, 16:48, 0:128]
        image_crop_2 = image[:, :, 48:80, 0:128]
        image_crop_3 = image[:, :, 80:112, 0:128]

        # padding

        image_pad_1 = F.pad(image_crop_1, (0, 0, 16, 80))
        image_pad_2 = F.pad(image_crop_2, (0, 0, 48, 48))
        image_pad_3 = F.pad(image_crop_3, (0, 0, 80, 16))
        return [image_pad_1, image_pad_2, image_pad_3]
