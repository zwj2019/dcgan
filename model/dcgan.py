import os
import argparse

import torch
from torchvision import datasets, transforms

import pytorch_lightning as pl
from model.basic_model import Generator, Discriminator

class DCGAN(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d = 'cpu'
        if args.ngpu > 0:
            assert torch.cuda.is_available()
            self.d = 'cuda'

        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()


    def init_generator(self):
        generator = Generator(self.args.ngpu, self.args.nz, self.args.nc, self.args.ngf)
        generator.apply(self.weights_init)
        print(generator)
        return generator
    
    def init_discriminator(self):
        discriminator = Discriminator(self.args.ngpu, self.args.nc, self.args.ngf)
        discriminator.apply(self.weights_init)
        print(discriminator)
        return discriminator

    # custom weights initialization called on netG and netD
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        return self.generator(z)
    
    def generator_loss(self, b_size):
        noise = torch.randn(b_size, self.args.nz, 1, 1, device=self.d)
        label = torch.ones((b_size,), device=self.d)

        fake = self.generator(noise)
        output = self.discriminator(fake).view(-1)
        loss = torch.nn.BCELoss()(output, label)
        return loss

    def discriminator_loss(self, x):
        b_size = x.size(0)
        real_label = torch.ones((b_size,), dtype=torch.float, device=self.d)
        fake_label = torch.zeros((b_size,), dtype=torch.float, device=self.d)
        criterion = torch.nn.BCELoss()

        # forward pass real batch through D
        output = self.discriminator(x).view(-1)
        real_loss = criterion(output, real_label)

        # forward pass fake batch throgh D
        noise = torch.randn(b_size, self.args.nz, 1, 1, device=self.d)
        fake = self.generator(noise)
        
        fake_loss = criterion(output, fake_label)

        return real_loss, fake_loss
    
    def generator_step(self, b_size):
        g_loss = self.generator_loss(b_size)
        result = pl.TrainResult(minimize=g_loss, checkpoint_on=g_loss)
        result.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return result
    
    def discriminator_step(self, x):
        d_real_loss, d_fake_loss = self.discriminator_loss(x)
        d_loss = d_real_loss + d_fake_loss
        result = pl.TrainResult(minimize=d_loss)
        result.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return result
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(x.size(0))

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(x)

        return result

    def configure_optimizers(self):
        beta1 = self.args.beta1

        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.d_lr, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.g_lr, betas=(beta1, 0.999))
        return [optimizerD, optimizerG], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--g_lr', help="Learning rate for Generator", type=float, default=0.0002)
        parser.add_argument('--d_lr', help="Learning rate for Discriminator", type=float, default=0.0002)
        parser.add_argument('--ndf', help="Depth of feature maps propagated through the discriminator", type=int, default=64)
        parser.add_argument('--ngf', help="Depth of feature maps carried through the generator", type=int, default=64)
        parser.add_argument('--nz', help="Length of latent vector", type=int, default=100)
        parser.add_argument('--beta1', help="beta1 hyperparameter for Adam optimizers", type=float, default=0.5)

        return parser
    