import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .attention import Block
from .model import SlotAttention, spatial_broadcast, SoftPositionEmbed, unstack_and_split

class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, out_channels+1)
        
        self.transformer_encoder = Block(dim=512, num_heads=16, mlp_ratio=4., 
                                         qkv_bias=False, drop=0., attn_drop=0.,
                                         act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.iters = 5
        self.num_slots = 12
        self.encoder_dims = 512
        self.decoder_initial_size = [8, 14]
        self.transformer_decoder = SlotAttention(iters=self.iters,
                                                 num_slots=self.num_slots,
                                                 encoder_dims=self.encoder_dims,
                                                 hidden_dim=self.encoder_dims)
        self.decoder_pos = SoftPositionEmbed(self.encoder_dims, self.decoder_initial_size)
    def pick_middle_repeat(self, x):
        ####Input x has shape: B*t, C, H, W
        x = einops.rearrange(x, '(b t) c h w -> b t c h w', t=5)
        x = x[:, 1:4]
        x = einops.repeat(x, 'b t c h w -> b (r t) c h w', r=4)
        x = einops.rearrange(x, 'b s c h w -> (b s) c h w')
        return x
    def forward(self, x):
        ## input: 'image' has shape B, 5(T), C, H, W  
        ## output:
        ###### 'recon_flow' has shape B, 3, 2, C, H, W 
        ###### 'recons' has shape B, 3, 2, 2(num_slot), C, H, W 
        ###### 'masks' has shape B, 3, 2, 2(num_slot), 1, H, W 
        B = x.shape[0]
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) ####B*t, 512, 8, 14

        #############slot attention to encode temporal information    
        x_flatten = einops.rearrange(x5, '(b t) c h w -> b (t h w) c', t=5)
        x_flatten = self.transformer_encoder(x_flatten)
        slots = self.transformer_decoder(x_flatten) #B, num_slot(12), C
#         
        #############cross attention     
        
        # Spatial broadcast decoder.
        x5 = spatial_broadcast(slots, [8, 14]) #192 = 16 * 12
#         slot_t = einops.rearrange(slot, 'b (t p s) c -> b t p s c', t=3, p=2)
        # `x` has shape: [batch_size*num_slots, height_init, width_init, slot_size].
        x5 = self.decoder_pos(x5)
        x5 = einops.rearrange(x5, 'b_n h w c -> b_n c h w')
        
        x = self.up1(x5, self.pick_middle_repeat(x4))
        x = self.up2(x, self.pick_middle_repeat(x3))
        x = self.up3(x, self.pick_middle_repeat(x2))
        x = self.up4(x, self.pick_middle_repeat(x1))

        x = self.outc(x)
        # `x` has shape: [batch_size*num_slots, num_channels+1, height, width].
        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=B, num_channels=self.out_channels)
        # `recons` has shape: [batch_size, num_slots, num_channels, height, width].
        # `masks` has shape: [batch_size, num_slots, 1, height, width].
        recons = einops.rearrange(recons, 'b (t p s) c h w -> b t p s c h w', t=3, p=2)
        masks = einops.rearrange(masks, 'b (t p s) c h w -> b t p s c h w', t=3, p=2)
        # Normalize alpha masks over slots.
        masks = torch.softmax(masks, axis=3)

        recon_combined = torch.sum(recons * masks, axis=3)  # Recombine image.
        # `recon_combined` has shape: [batch_size, temporal, 2(t), num_channels, height, width].
        return recon_combined, recons, masks, slots

    
if __name__ == '__main__':
    model = UNet(3, 3).cuda()
    data_input = torch.randn(16, 5, 3, 128, 224).cuda()
    recon_combined, recons, masks, slots = model(data_input)
    
