import torch
import torch.nn as nn
from models.axialdw import AxialDW
from models.vssblock import VSSBlock
from models.bottleneck import BottleneckPCAPSA
from models.bottleneck import CSAG
from models.bottleneck import CBAM
from models.selective_kernel import SKConv_7


class ResMambaBlock(nn.Module):
    def __init__(self, in_c, k_size = 3):
      super().__init__()
      self.in_c = in_c
      self.conv = nn.Conv2d(in_c, in_c, k_size, stride=1, padding='same', dilation=1, groups=in_c, bias=True, padding_mode='zeros')
      self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
      self.act = nn.LeakyReLU(negative_slope=0.01)
      self.block = VSSBlock(hidden_dim = in_c)
      self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):

      skip = x

      x = self.conv(x)
      x = x.permute(0, 2, 3, 1)
      x = self.block(x)
      x = x.permute(0, 3, 1, 2)
      x = self.act(self.ins_norm(x))
      return x + skip * self.scale
    

class EncoderBlock_Mamba(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c,mixer_kernel=(3,3)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = mixer_kernel )
        self.pw= nn.Conv2d(in_c, out_c, kernel_size=1, padding = 'same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.resmamba = ResMambaBlock(in_c)
        self.down = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.resmamba(x)
        skip = self.act(self.bn(self.pw(self.dw(x))))
        x = self.down(skip)

        return x, skip
    

class EncoderBlock_Axial(nn.Module):
      def __init__(self, in_c, out_c, mixer_kernel = (7,7)):
          super().__init__()
          self.dw = AxialDW(in_c, mixer_kernel = mixer_kernel )
          self.bn = nn.BatchNorm2d(in_c)
          self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
          self.down = nn.MaxPool2d((2,2))
          self.act = nn.ReLU()

      def forward(self, x):
          skip = self.bn(self.dw(x))
          skip = self.pw(skip)
          x = self.act(self.down(skip))
          return x, skip
      
class DecoderBlock_Axial(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.pw = nn.Conv2d(in_c, out_c, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.ReLU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
    def forward(self, x):
        x = self.act(self.pw2(self.dw(self.bn2(self.pw(x)))))


        return x
    

class DecoderBlock_Mamba(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (3, 3))
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=1, padding = 'same')
        self.bn2 = nn.BatchNorm2d(out_c)
        self.resmamba = ResMambaBlock(out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn2(self.conv2(self.dw(x))))

        x = self.resmamba(x)

        return x
    

class ADC_MambaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsize = nn.Upsample(scale_factor = 2)
        self.pw_in = nn.Conv2d(3, 16, kernel_size=1)
        self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1 ,L=32)
        """Encoder"""
        self.e1 = EncoderBlock_Mamba(16, 32)
        self.e2 = EncoderBlock_Mamba(32, 64)
        self.e3 = EncoderBlock_Mamba(64, 128)
        self.e4 = EncoderBlock_Axial(128, 256)
        self.e5 = EncoderBlock_Axial(256, 512)

        '''CSAG'''
        self.c1=CSAG(32)
        self.c2=CSAG(64)
        self.c3=CSAG(128)
        self.c4=CSAG(256)
        self.c5=CSAG(512)

        """Skip connection"""
        self.s1 = CBAM(gate_channels = 32)
        self.s2 = CBAM(gate_channels = 64)
        self.s3 = CBAM(gate_channels = 128)
        self.s4 = CBAM(gate_channels = 256)
        self.s5 = CBAM(gate_channels = 512)

        """Bottle Neck"""
        self.b52=BottleneckPCAPSA(512)

        """Decoder"""
        self.d5 = DecoderBlock_Axial(512, 512, 256)
        self.d4 = DecoderBlock_Axial(256, 256, 128)
        self.d3 = DecoderBlock_Mamba(128, 128, 64)
        self.d2 = DecoderBlock_Mamba(64, 64, 32)
        self.d1 = DecoderBlock_Mamba(32, 32, 16)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)
    def forward(self, x):
        """Encoder"""
        x = self.pw_in(x)
        x = self.sk_in(x)
        x, skip1 = self.e1(x)

        x, skip2 = self.e2(x)

        x, skip3 = self.e3(x)

        x, skip4 = self.e4(x)

        x, skip5 = self.e5(x)
        """BottleNeck"""
        x=self.b52(x)
        """Skip5+Decoder5"""
        x=self.upsize(x)
        skip5 = self.c5(x,skip5)
        skip5 = self.s5(skip5)
        x = self.d5(skip5)
        """Skip4+Decoder4"""
        x=self.upsize(x)
        skip4 = self.c4(x,skip4)
        skip4 = self.s4(skip4)
        x = self.d4(skip4)
        """Skip3+Decoder3"""
        x=self.upsize(x)
        skip3 = self.c3(x,skip3)
        skip3 = self.s3(skip3)
        x = self.d3(skip3)
        """Skip2+Decoder2"""
        x=self.upsize(x)
        skip2 = self.c2(x,skip2)
        skip2 = self.s2(skip2)
        x = self.d2(skip2)
        """Skip1+Decoder1"""
        x=self.upsize(x)
        skip1 = self.c1(x,skip1)
        skip1 = self.s1(skip1)
        x = self.d1(skip1)

        x = self.conv_out(x)
        return x