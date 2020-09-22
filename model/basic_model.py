import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz, nc, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(type(x), x.size())
        return self.main(x)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention)
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class SelfAttention2(nn.Module):
    
    def __init__(self, in_dim):
        super(SelfAttention2, self).__init__()
        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1)
    
    def flatten(self, x):
        assert len(x.size()) == 4
        B, C, W, H = x.size()
        
        x = x.view(B, -1, W * H)
        return x

    def forward(self, x):
        # x : B, C, W, H
        B, C, W, H = x.size()
        q = self.query(x)
        q = self.max_pool1(q)
        q = self.flatten(q) # B, C//8, W * H / 4

        k = self.key(x)
        k = self.flatten(k) # B, C//8, W * H

        v = self.value(x)
        v = self.max_pool2(v)
        v = self.flatten(v) # B, C//2, W * H / 4

        atte = torch.bmm(q.permute(0, 2, 1), k)
        atte = self.softmax(atte) # B, W * H, W * H / 4

        out = torch.bmm(v, atte)
        out = out.view(B, -1, W, H)
        out = self.output(out) # B, C//2, W * H
        
        return x + self.gamma * out


if __name__ == '__main__':
    # Test Generator and Discriminator
    # x = torch.randn(128, 100, 1, 1)
    # g = Generator(1, 100, 3, 64)
    # d = Discriminator(1, 3, 64)
    # output = g(x)
    # d_output = d(output)
    # print(d_output.size())
    # print(output.size())

    # Test self-attention
    atte = SelfAttention2(512)
    x = torch.randn(13, 512, 4, 4)
    o = atte(x)
    print(o.size())
