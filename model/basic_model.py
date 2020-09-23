import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SimpleSelfAttention, self).__init__()
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
        return out


class SelfAttention(nn.Module):
    
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
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


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, attention=None):
        super(Generator, self).__init__()
        if not attention is None:
            assert isinstance(attention, str) and attention in ["simple", "normal"], '`attention` must be "simple" or "normal"'
        
        self.ngfx8x4x4 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        self.ngfx4x8x8 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.ngfx2x16x16 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.ngfx32x32 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        self.ncx64x64 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        if attention == "simple":
            self.att1 = SimpleSelfAttention(ngf * 2)
            self.att2 = SimpleSelfAttention(ngf)
        elif attention == "normal":
            self.att1 = SelfAttention(ngf * 2)
            self.att2 = SelfAttention(ngf)
        else:
            self.att1 = self.att2 = None
        
        
    def forward(self, x):
        x = self.ngfx8x4x4(x)
        x = self.ngfx4x8x8(x)
        x = self.ngfx2x16x16(x)
        if not self.att1 is None:
            x = self.att1(x)
        x = self.ngfx32x32(x)
        if not self.att2 is None:
            x = self.att2(x)
        out = self.ncx64x64(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, attention=None):
        super(Discriminator, self).__init__()
        if not attention is None:
            assert isinstance(attention, str) and attention in ["simple", "normal"], '`attention` must be "simple" or "normal"'
        
        self.ndfx32x32 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ndfx2x16x16 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ndfx4x8x8 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.ndfx8x4x4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.last = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        if attention == "simple":
            self.att1 = SimpleSelfAttention(ndf * 4)
            self.att2 = SimpleSelfAttention(ndf * 8)
        elif attention == "normal":
            self.att1 = SelfAttention(ndf * 4)
            self.att2 = SelfAttention(ndf * 8)
        else:
            self.att1 = self.att2 = None
        
    def forward(self, x):
        x = self.ndfx32x32(x)
        x = self.ndfx2x16x16(x)
        x = self.ndfx4x8x8(x)
        if not self.att1 is None:
            x = self.att1(x)
        x = self.ndfx8x4x4(x)
        if not self.att2 is None:
            x = self.att2(x)
        out = self.last(x)
        return out


if __name__ == '__main__':
    x = torch.randn(13, 100, 1, 1)
    g = Generator(100, 3, 64, attention="simple")
    d = Discriminator(3, 64, attention="simple")

    fake = g(x)
    print(fake.size())
    dis = d(fake)
    print(dis.size())