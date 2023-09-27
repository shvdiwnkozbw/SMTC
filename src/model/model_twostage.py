import torch
import einops
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .attention import Block, InstanceNorm1d, BasicConv2d, CrossBlock, BasicBlock, Conv2Plus1D, MEBlock
from .dconv import DeformableConv2d
from .vision_transformer import vit_small, vit_base

def build_grid(resolution):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [res for res in resolution] + [-1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.tensor(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)


class SoftPositionEmbed(nn.Module):
  """Adds soft (include spatio-temporal) positional embedding with learnable projection."""
  def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.proj = nn.Linear(len(resolution)*2, hidden_size)
        self.grid = build_grid(resolution)
  def forward(self, inputs):
        return inputs + self.proj(self.grid)


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = torch.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    grid = einops.repeat(slots, 'b_n i j d -> b_n (tilei i) (tilej j) d', tilei=resolution[0], tilej=resolution[1])
    # `grid` has shape: [batch_size*num_slots, height, width, slot_size].
    return grid


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = einops.rearrange(x, '(b s) c h w -> b s c h w', b=batch_size)
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
    return channels, masks


class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, num_instances=4, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()
        
        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.num_instances = num_instances
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, 1, encoder_dims))
        self.slots_sigma = nn.Parameter(torch.zeros(1, num_slots, 1, encoder_dims))
        nn.init.xavier_uniform_(self.slots_sigma)

        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

        # Slot update functions.
        # self.gru = nn.GRUCell(encoder_dims, encoder_dims)

    def forward(self, inputs, bs, weight, init_slots=None, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        # inputs = inputs + self.order_embedding
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = self.num_slots
        n_p = self.num_instances

        # random slots initialization,
        semantic_slots = self.slots_mu.expand(b, -1, -1, -1).squeeze(2) # b ns c
        mu = self.slots_mu.expand(b, -1, n_p, -1)
        sigma = self.slots_sigma.exp().expand(b, -1, n_p, -1)
        instance_slots = torch.normal(mu, sigma) # b ns np c

        # Multiple rounds of semantic slot attention.
        slots = semantic_slots
        for t in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            # slots = self.gru(
            #     updates.reshape(-1, d),
            #     slots_prev.reshape(-1, d)
            # )
            # slots = slots.reshape(b, -1, d)
            # slots = slots + self.mlp(self.norm_pre_ff(slots))
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            if t == self.iters-2:
                slots = slots.detach() - semantic_slots.detach() + semantic_slots
        semantic_slots = slots # b ns c
        semantic_attn = dots # b ns hw

        slots = instance_slots
        for t in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            dots = torch.einsum('bkid,bjd->bkij', q, k) * self.scale
            attn = dots.softmax(dim=2) + self.eps
            attn = attn * torch.softmax(semantic_attn, dim=1).unsqueeze(2)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bkij->bkid', v, attn)
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            if t == self.iters-2:
                slots = slots.detach() - instance_slots.detach() + instance_slots

        instance_slots = slots
        instance_attn = dots
        # q = self.project_q(self.norm_slots(slots))
        # dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        # attn = dots.softmax(dim=1) + self.eps

        return semantic_slots, semantic_attn, instance_slots, instance_attn


class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""
    def __init__(self, resolution, 
                       num_slots, 
                       num_instances,
                       in_channels=3, 
                       out_channels=3, 
                       hid_dim=32,
                       iters=5, 
                       path_drop=0.1,
                       attn_drop_t=0.4,
                       attn_drop_f=0.2,
                       num_frames=7,
                       teacher=False,
                       dino_path='/home/ma-user/work/shuangrui/01_feature_warp/dino_deitsmall16_pretrain.pth'
                ):
        """Builds the Slot Attention-based Auto-encoder.
        Args:
            resolution: Tuple of integers specifying width and height of input image
            num_slots: Number of slots in Slot Attention.
            iters: Number of iterations in Slot Attention.
        """
        super(SlotAttentionAutoEncoder, self).__init__()

        self.iters = iters
        self.num_slots = num_slots
        self.num_instances = num_instances
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T = num_frames
        
        self.encoder_dims = 384
        self.hid_dims = 128
        self.encoder = vit_small(16)
        self.encoder.load_state_dict(torch.load(dino_path), strict=False)
        
        self.down_time = 16
        self.end_size = (resolution[0] // self.down_time, resolution[1] // self.down_time)
        
        self.transport_proj = nn.Conv1d(32*32, self.encoder_dims, kernel_size=1, padding=0)
        self.teacher = teacher
        self.mlp_slot = nn.Sequential(
            nn.Linear(self.encoder_dims, 2*self.encoder_dims), ##TODO: test mlp_ratio
            nn.ReLU(inplace=True),
            nn.Linear(2*self.encoder_dims, self.encoder_dims)
        )
        self.slot_attention = SlotAttention(
            iters=self.iters,
            num_slots=self.num_slots,
            num_instances=self.num_instances,
            encoder_dims=self.encoder_dims,
            hidden_dim=self.encoder_dims)

    def forward(self, image, weight=0.01, p_s=None):
        ## input: 'image' has shape B, 5(T), C, H, W  
        ##        'p_s' has shape B, 4 (random sample)
        ## output:
        ###### 'sudo_mask' has shape B, 7, (H, W) 
        ###### 'slots' has shape B, (7, 2), 2(num_slot), C
        ###### 'motion_mask' has shape B, (7, 2), 2(num_slot), (H, W) 
        ###### 'slot_warp' has shape B (7, 2), C
        ###### 'x_warp' has shape B, (7, 2), C
        
        # Convolutional encoder with position embedding.
        bs = image.shape[0]
        image_t = einops.rearrange(image, 'b t c h w -> (b t) c h w')
        x, cls_token, attn, k = self.encoder(image_t)  # CNN Backbone/ DINO backbone
        x = einops.rearrange(x, '(b t) c h w -> b t c (h w)', t=self.T) ##spatial_flatten
        k = einops.rearrange(k, '(b t) c h w -> b t c (h w)', t=self.T)
        frame = einops.rearrange(k, 'b t c hw -> b t hw c')
        correlation_map = self.calculate_transport(k, torch.roll(k, shifts=1, dims=1))
        k = k + correlation_map
        k = einops.rearrange(k, 'b t c hw -> b t hw c')

        sudo_mask = attn[:, :, 0, 1:].mean(dim=1)
        sudo_mask = sudo_mask / sudo_mask.sum(dim=-1, keepdim=True)
        sudo_mask = einops.rearrange(sudo_mask, '(b t) hw -> b t hw', t=self.T)
        
        slots, motion_mask, instance, instance_mask = self.decode(x, k, weight, ts=self.T)
        if p_s is None:
            motion_mask = F.softmax(motion_mask, dim=-2) + 1e-8
            return sudo_mask, x, motion_mask, self.end_size[0]
        else:
            warp_instance = self.mlp_slot(instance)
            warp_instance = F.normalize(warp_instance, dim=-1, p=2)
            return motion_mask, frame, slots, instance, warp_instance
    
    def calculate_transport(self, x_start, x_end):
        b, t, c, n = x_start.shape
        x_start = x_start.view(b*t, c, n)
        x_end = x_end.view(b*t, c, n)
        correlation = torch.einsum('bcn,bcm->bnm', x_end, x_start)
        correlation = self.transport_proj(correlation)
        return correlation.view(b, t, self.encoder_dims, n)

    def decode(self, x, k, weight, ts=7):
        # x bs t hw c
        # slot bs t s c
        # motion_mask bs t s hw
        bs = x.shape[0]
        x = einops.rearrange(x, 'b t hw c -> (b t) hw c')
        k = einops.rearrange(k, 'b t hw c -> (b t) hw c')
        slots, motion_mask, instance, instance_mask = self.slot_attention(k, bs, weight) 
        slots = einops.rearrange(slots, '(b t) s c -> b t s c', b=bs)
        motion_mask = einops.rearrange(motion_mask, '(b t) s hw -> b t s hw', b=bs)
        instance = einops.rearrange(instance, '(b t) s p c -> b t s p c', b=bs)
        instance_mask = einops.rearrange(instance_mask, '(b t) s p hw -> b t s p hw', b=bs)
        return slots, motion_mask, instance, instance_mask


if __name__ == "__main__":
    model = SlotAttentionAutoEncoder(resolution=(192, 384), 
                       num_slots=2, 
                       in_channels=3, 
                       out_channels=3, 
                       hid_dim=32,
                       iters=5, 
                       path_drop=0.1,
                       attn_drop_t=0.4,
                       num_frames=7)
    model.to('cuda')
    input = torch.randn(2, 7, 3, 192, 384)
    model(input.to('cuda'))
