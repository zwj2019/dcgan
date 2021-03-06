import os
import argparse

import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from model.basic_model import Generator, Discriminator


class DCGAN(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()

    def init_generator(self):
        generator = Generator(self.args.nz, self.args.nc, self.args.ngf, attention=self.args.attention)
        generator.apply(self.weights_init)
        print(generator)
        return generator
    
    def init_discriminator(self):
        discriminator = Discriminator(self.args.nc, self.args.ngf, attention=self.args.attention)
        discriminator.apply(self.weights_init)
        print(discriminator)
        return discriminator

    # custom weights initialization called on Generator and Discriminator
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
        noise = torch.randn(b_size, self.args.nz, 1, 1, device=self.device)
        label = torch.ones((b_size,), dtype=torch.float, device=self.device)

        fake = self.generator(noise)
        output = self.discriminator(fake).view(-1)
        loss = F.binary_cross_entropy(output, label)
        return loss

    def discriminator_loss(self, x):
        b_size = x.size(0)
        real_label = torch.ones((b_size,), dtype=torch.float, device=self.device)
        fake_label = torch.zeros((b_size,), dtype=torch.float, device=self.device)

        # forward pass real batch through D
        output = self.discriminator(x).view(-1)
        real_loss = F.binary_cross_entropy(output, real_label)

        # forward pass fake batch throgh D
        noise = torch.randn(b_size, self.args.nz, 1, 1, device=self.device)
        fake = self.generator(noise)
        output = self.discriminator(fake).view(-1)

        fake_loss = F.binary_cross_entropy(output, fake_label)

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
            result = self.discriminator_step(x)
        # train discriminator
        if optimizer_idx == 1:
            result = self.generator_step(x.size(0))
        
        # Generate some sample images
        if (batch_idx % 500 == 0) or ((self.current_epoch == self.trainer.max_epochs) and 
                    batch_idx == len(self.trainer.train_dataloader) - 1):
                    output_path = self.args.output

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    fake_iamge = None
                    with torch.no_grad():
                        random_noise = torch.randn(16, 100, 1, 1, device=self.device)
                        fake_image = self.generator(random_noise).detach().cpu()
                    
                    assert not fake_image is None
                    img_tensor_list = [tensor for tensor in fake_image]
                    img_list = [img.numpy().transpose((1, 2, 0)) for img in img_tensor_list]
                    plt.figure()
                    for i in range(len(img_list)):
                        plt.subplot(4, 4, i + 1)
                        plt.imshow(img_list[i] * 0.5 + 0.5)
                        plt.xticks([])
                        plt.yticks([])
                    plt.savefig(os.path.join(output_path, 'aaattn_epoch-%d-batch_idx-%d.jpg' % (self.current_epoch, batch_idx)))
                    plt.close()
        return result

    def configure_optimizers(self):
        beta1 = self.args.beta1

        optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.d_lr, betas=(beta1, 0.999))
        optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.args.g_lr, betas=(beta1, 0.999))
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
        parser.add_argument('--attention', help="Type of `attention`", choices=["simple", "normal"] ,type=str, default=None)


        return parser
    