import torch
import torch.nn as nn
from einops import reduce

# Priority Channel Attention (PCA)
class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute the channel-wise mean of the input
        c = reduce(x, 'b c w h -> b c', 'mean')

        # Apply depthwise convolution
        x = self.dw(x)

        # Compute the channel-wise mean after convolution
        c_ = reduce(x, 'b c w h -> b c', 'mean')

        # Compute the attention scores
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(2).unsqueeze(3)  # Shape [batch_size, channels, 1, 1]
        return x * att_score  # Broadcasting to match the dimensions


# Priority Spatial Attention (PSA)

class PSA(nn.Module):
    def __init__(self, dim,H=8,W=8):

        super().__init__()

        # self.dw = nn.Conv2d(dim, dim, kernel_size = 9, groups=dim, padding="same")
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.pw_h = nn.Conv2d(H, H, (1, 1))
        self.pw_w = nn.Conv2d(W, W, (1, 1))
        self.prob = nn.Softmax2d()
    def forward(self, x):

        s = reduce(x , 'b c w h -> b w h', 'mean')

        x1 = self.pw(x)
        x_h = self.pw_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # x_w = self.pw_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = self.pw_w(x.permute(0,2,3,1)).permute(0,3,1,2)
        s_ = reduce(x , 'b c w h -> b w h', 'mean')
        s_h = reduce(x_h , 'b c w h -> b w h', 'mean')
        s_w = reduce(x_w , 'b c w h -> b w h', 'mean')

        raise_sp = self.prob(s_ - s)
        raise_h = self.prob(s_h - s)
        raise_w = self.prob(s_w - s)

        att_score = torch.sigmoid(s_*(1 + raise_sp)+s_h*(1 + raise_h)+s_w*(1 + raise_w))
        att_score = att_score.unsqueeze(1)
        return x * att_score

class BottleneckPCAPSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pca = PCA(dim)
        self.psa = PSA(dim)
    def forward(self, x):
        # Apply PCA with residual connection
        pca_out=self.pca(x)
        # Apply PSA
        psa_out = self.psa(pca_out)

        return psa_out
