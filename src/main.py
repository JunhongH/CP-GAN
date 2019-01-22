import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from  utils import utils
from models.transformer_net import TransformerNet
import models.networks as networks
from dataset.datasets import TrainDataset, TestDataset
from tensorboardX import SummaryWriter
from models.light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
import torchvision.utils as vutils

from models.GAN_model import GAN


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args=utils.get_gpu_list(args)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # define transforms of cartoon face images 

    transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    # define transforms of real face images 

    style_transform = transforms.Compose([
        transforms.Scale(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = TrainDataset(args.dataset,args.style_dataset,transform,style_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,num_workers=4, shuffle=True)

    gan=GAN(args)

    norm_loss = torch.nn.MSELoss()



    # content network

    content_network = LightCNN_9Layers(use_fc=args.use_fc,requires_grad=False)
    if args.cuda:
        content_network = content_network.cuda()

    # load pretrained content network

    checkpoint = torch.load('./src/models/LightCNN_9Layers_checkpoint.pth.tar')
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = content_network.state_dict()
    for k, v in state_dict.items():
        name = k[7:]
        if 'fc2' not in name:
            new_state_dict[name] = v
    content_network.load_state_dict(new_state_dict)
        

    # test function
    test=Test(args)
    save_address=os.path.join(args.save_model_dir,"experiment_{}".format(args.num_experiment))
    # send data to tensorboad 
    writer=SummaryWriter(save_address+"/logs/")
    count = 0+args.start_epoch*len(train_dataset)
    agg_content_loss = 0.
    agg_style_loss = 0.
    agg_G_loss=0.
    agg_identity_loss=0.
    agg_style_local_loss=0.
    agg_G_local_loss=0.

    # start training

    for e in range(args.start_epoch,args.start_epoch+args.epochs):
        gan.netG.train()
        
        for batch_id_1, (x, style) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch

            # total batch_id

            batch_id=batch_id_1+e*len(train_loader)
            # current epoch count
            current_epoch_count=count-e*len(train_dataset)

            x = Variable(x)
            style=Variable(style)

            if args.cuda:
                x = x.cuda()
                style=style.cuda()

            y = gan.netG(x)

            if args.identity_weight>0:
                idt_style=gan.netG(style)
                identity_loss=args.identity_weight * gan.identity_criterion(idt_style , style )
            else:
                identity_loss=0

            # normalize images 

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            style=utils.normalize_batch(style)

            features_y = content_network(y)
            features_x = content_network(x)

            # content loss used to keep content

            content_loss= args.content_weight * norm_loss(features_y, features_x)

            # cropping and padding
            style_pathes=gan.crop_and_pad(style)
            y_pathes=gan.crop_and_pad(y)

            G_local_loss=0.
            style_local_loss = 0.
            local_weight=[1,1,1]
            for index in range(3):

                # G local loss
               
                G_local_loss+=gan.update_G_local(y_pathes[index])*local_weight[index]

                # D local loss 

                style_local_loss+=gan.update_D_local(style_pathes[index],y_pathes[index])*local_weight[index]

            # update local G loss

            gan.optimizer_G.zero_grad()
            G_local_loss.backward(retain_graph=True)
            gan.optimizer_G.step() 

            # update local D loss

            gan.optimizer_D_local.zero_grad()
            style_local_loss.backward()
            gan.optimizer_D_local.step()

            # update global G loss

            gan_loss=gan.update_G(y)
            gan.optimizer_G.zero_grad() 

            G_loss=0
            G_loss=gan_loss + content_loss + identity_loss 
            G_loss.backward()

            gan.optimizer_G.step()

            # update global D loss

            gan.optimizer_D.zero_grad()
            style_loss = 0.
            style_loss+=gan.update_D(style , y)
            style_loss.backward()
            gan.optimizer_D.step()

            
            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]
            agg_style_local_loss+=style_local_loss.data[0]
            agg_G_loss += gan_loss.data[0]
            agg_G_local_loss+=G_local_loss.data[0]
            
            if args.identity_weight>0:
                agg_identity_loss+=identity_loss.data[0]
            else:
                agg_identity_loss+=0

            # log intervals 

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tD: {:.6f}\tG: {:.6f}\tG_local: {:.6f}\tD_local: {:.6f}\tidt: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, current_epoch_count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  agg_G_loss / (batch_id + 1),
                                  agg_G_local_loss / (batch_id + 1),
                                  agg_style_local_loss / (batch_id + 1),
                                  agg_identity_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss+agg_G_loss+agg_identity_loss) / (batch_id + 1)
                )
                with open(os.path.join(save_address,"log.txt"), "a") as log_file:
                    log_file.write('%s\n' % mesg)
                print(mesg)

                # checkpoint

                if args.checkpoint:
                    writer.add_scalars("loss",{
                                 'D_loss':agg_style_loss / (batch_id + 1),
                                 'G_loss':agg_G_loss / (batch_id + 1),
                                 'content_loss':agg_content_loss / (batch_id + 1),
                                 'identity_loss':agg_identity_loss/(batch_id+1),
                                 'G_local_loss':agg_G_local_loss / (batch_id + 1),
                                 'D_local_loss':agg_style_local_loss / (batch_id + 1),
                                  
                                 },count)

               
            # save models
            if (batch_id + 1) % (4*args.log_interval) == 0:
                gan.netG.eval()
                output_name="Epoch_"+str(e)  + "_iters_" +str(current_epoch_count)+"_G" +".jpg"
                output_name=os.path.join(save_address,output_name)
                test.run_test(gan.netG,output_name)
                if args.cuda:
                    gan.netG.cpu()
                    gan.netD.cpu()
                save_model_filename = "Epoch_"+str(e)  + "_iters_" +str(current_epoch_count)+"_G" +".model"
                save_model_path = os.path.join(save_address,save_model_filename)
                torch.save(gan.netG.state_dict(), save_model_path)

                save_model_filename_D = "Epoch_"+str(e)  + "_iters_" +str(current_epoch_count)+"_D" +".model"
                save_model_path_D = os.path.join(save_address,save_model_filename_D)
                torch.save(gan.netD.state_dict(), save_model_path_D)
                print("\nCheckpoint, trained model saved at", save_model_path)

                if args.cuda:
                    gan.netG.cuda()
                    gan.netD.cuda()
                gan.netG.train()


            
    # save last model
    gan.netG.eval()
    if args.cuda:
        gan.netG.cpu()
    save_model_filename = "Epoch_"+str(e)  + "_iters_" +str(current_epoch_count)+"_G" +".model"
    save_model_path = os.path.join(save_address,save_model_filename)
    torch.save(gan.netG.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)



# test in training phase

class Test():
    def __init__(self,args):
        test_transform = transforms.Compose([
                                    transforms.Scale(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))
                                                ])
        test_dataset=TestDataset(args.test_dataset,test_transform)
        self.test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,num_workers=4, shuffle=False)

        self.args=args
    def run_test(self,net,output_image):
        output_list=[]
        for x in self.test_loader:

            if self.args.cuda:
                x=x.cuda()
            x=Variable(x,volatile=True)
            output = net(x)

            output = output.data
            output_list.append(output)
        output_total=torch.cat(output_list,0)
        vutils.save_image(output_total, output_image, nrow=3, normalize=True)
        return output_total[0]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# use to test dataset

def stylize(args):
    img_list=os.listdir(args.content_image)
    epoch_name=os.path.basename(args.model).split('.')[0]
    experiment_name=os.path.dirname(args.output_image)
    if not os.path.exists(experiment_name):
        os.system('mkdir {}'.format(experiment_name))
    if not os.path.exists(args.output_image):
        os.system('mkdir {}'.format(args.output_image))
    for img in img_list:
        if is_image_file(img):
            content_image = utils.load_image(os.path.join(args.content_image,img), scale=args.content_scale)
            content_image=content_image.convert('RGB')
            content_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0)
            if args.cuda:
                content_image = content_image.cuda()
            content_image = Variable(content_image, volatile=True)

            style_model = TransformerNet()
            style_model.load_state_dict(torch.load(args.model))
            if args.cuda:
                style_model.cuda()
            output = style_model(content_image)
            if args.cuda:
                output = output.cpu()
                content_image=content_image.cpu()
            output_data = output.data[0]
            content_image_data=content_image.data[0]
            output_data=torch.cat([content_image_data,output_data],2)
            output_name=os.path.join(args.output_image,epoch_name+"_result_"+img)
            utils.save_image(output_name, output_data)



def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument('--test_batch_size', type=int, default=1, help='the batch size of test dataset')
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style_dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--test_dataset", type=str, required=True,
                                  help="which metric graient loss uses")  
    train_arg_parser.add_argument('--content-weight', type=float, default=1, help='# of discrim filters in first conv layer')

    train_arg_parser.add_argument("--save-model-dir", type=str, default='checkpoint/',
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--num_experiment", type=int, default=0,
                                  help="number of experiment")
    train_arg_parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    train_arg_parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    train_arg_parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    train_arg_parser.add_argument('--gan_weight', type=float, default=1, help='# of input image channels')
    train_arg_parser.add_argument('--gan_local_weight', type=float, default=1, help='# of input image channels')
    train_arg_parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    train_arg_parser.add_argument('--ngf', type=int, default=64, help='# of discrim filters in first conv layer')
    train_arg_parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
    train_arg_parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    train_arg_parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    train_arg_parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
    train_arg_parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    train_arg_parser.add_argument("--checkpoint", action='store_true',help="set it to checkpoints online")
    train_arg_parser.add_argument('--identity_weight', type=float, default=1, help='the weight of target domain identity loss ')
    train_arg_parser.add_argument('--resume_netG', type=str, default='', help='reload netG model')
    train_arg_parser.add_argument('--resume_netD', type=str, default='', help='reload netD model')
    train_arg_parser.add_argument('--resume_netD_local', type=str, default='', help='reload netD_local model')
    train_arg_parser.add_argument('--start_epoch', type=int, default=0, help='then crop to this size')
    train_arg_parser.add_argument("--affine_state", type=bool,default=True, help="set the affine parameter of norm layer")
 
    
    train_arg_parser.add_argument('--use_fc', type=int, default=0, help='if use fc for light_cnn_9')


    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--content-size", type=int, default=512,
                                  help="the size of content image")
    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        utils.check_paths(args)
        save_test=utils.Save_text()
        save_test.save(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
