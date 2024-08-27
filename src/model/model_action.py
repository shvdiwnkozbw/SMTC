import torch
import einops
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .attention import Block, InstanceNorm1d, BasicConv2d, CrossBlock, BasicBlock, Conv2Plus1D, Attention, STBlock, SlotBlock
from .vision_transformer import vit_small, vit_base, vit_large
from . import clip

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

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8, background=False):
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
        self.scale = encoder_dims ** -0.5
        self.background = background
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)
        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)
        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, hidden_dim)
        self.project_k = nn.Linear(encoder_dims, hidden_dim)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)
        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, bs, weight, init_slots=None, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # learnable slots initializations
        if init_slots == None:
            init_slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))
            if self.background:
                init_slots = torch.cat([init_slots, torch.zeros_like(init_slots[:, -1:, :])], dim=1)
            slots = init_slots
        else:
            slots = init_slots

        # Multiple rounds of attention.
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
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            if t == self.iters-2:
                slots = slots.detach() - init_slots.detach() + init_slots

        return slots, dots


class LinearPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super(LinearPositionEmbedding, self).__init__()
        self.freq_bands = torch.linspace(0, dim//4-1, dim//4)

    def forward(self, correlation, NORMALIZE_FACOR=1/200, fusion='cat'):
        # correlation b h w m
        b, h, w, m = correlation.shape
        coord = torch.meshgrid(torch.arange(h), torch.arange(w))
        x = torch.stack(coord[::-1], dim=0).float().to(correlation.device)
        x = x.permute(1, 2, 0).contiguous()
        freq_bands = self.freq_bands.to(correlation.device)
        position_embed = torch.cat([torch.sin(3.14*x[..., -2:-1]*freq_bands*NORMALIZE_FACOR), 
                                    torch.cos(3.14*x[..., -2:-1]*freq_bands*NORMALIZE_FACOR), 
                                    torch.sin(3.14*x[..., -1:]*freq_bands*NORMALIZE_FACOR), 
                                    torch.cos(3.14*x[..., -1:]*freq_bands*NORMALIZE_FACOR)], dim=-1)
        position_embed = position_embed.unsqueeze(0).repeat(b, 1, 1, 1)
        return torch.cat([correlation, position_embed], dim=-1) if fusion=='cat' else correlation + position_embed


class SpaceTimeAttention(nn.Module):
    def __init__(self, num_frames, num_slots, encoder_dims, output_dim, attn_drop_t, path_drop, task):
        super(SpaceTimeAttention, self).__init__()
        self.T = num_frames
        self.num_slots = num_slots
        self.encoder_dims = encoder_dims
        self.task = task
        self.st_token = nn.Parameter(torch.rand(1, 1, self.encoder_dims) * .02)
        self.time_embed = nn.Parameter(torch.randn(1, num_frames, self.encoder_dims) * .02)
        self.slots_embed = nn.Parameter(torch.randn(1, self.num_slots+1, self.encoder_dims) * .02)
        self.pos_drop = nn.Dropout(path_drop)
        self.st_transformer = nn.ModuleList([STBlock(dim=self.encoder_dims, num_heads=8, drop=path_drop, attn_drop=attn_drop_t)
                                             for _ in range(3)])
        if task == 'object':
            self.mlp_token = nn.Sequential(
                nn.Linear(self.encoder_dims, 2*self.encoder_dims), ##TODO: test mlp_ratio
                nn.ReLU(inplace=True),
                nn.Linear(2*self.encoder_dims, output_dim)
            )
        elif task == 'action':
            self.cls_layer = nn.Linear(self.encoder_dims, 174)
        elif task == 'joint':
            self.mlp_token = nn.Sequential(
                nn.Linear(self.encoder_dims, 2*self.encoder_dims), ##TODO: test mlp_ratio
                nn.ReLU(inplace=True),
                nn.Linear(2*self.encoder_dims, output_dim)
            )
            self.cls_layer = nn.Linear(self.encoder_dims, 174)

    def forward(self, slots):
        slots_input = einops.rearrange(slots, 'b t s c -> (b t) s c')
        slots_input = torch.cat([self.st_token.expand(slots_input.shape[0], -1, -1), slots_input], dim=1)
        slots_input = slots_input + self.slots_embed
        slots_input = self.pos_drop(slots_input)
        st_token = slots_input[:slots.shape[0], 0].unsqueeze(1)
        slots_input = slots_input[:, 1:]
        slots_input = einops.rearrange(slots_input, '(b t) s c -> (b s) t c', t=self.T)
        slots_input = slots_input + self.time_embed
        slots_input = self.pos_drop(slots_input)
        slots_input = einops.rearrange(slots_input, '(b s) t c -> b (t s) c', s=self.num_slots)
        slots_input = torch.cat([st_token, slots_input], dim=1)
        for block in self.st_transformer:
            slots_input = block(slots_input, self.num_slots, self.T)
        st_token = slots_input[:, 0]
        if self.task == 'object':
            st_token = self.mlp_token(st_token)
            return st_token
        elif self.task == 'action':
            logit = self.cls_layer(st_token)
            return logit
        elif self.task == 'joint':
            logit = self.cls_layer(st_token)
            st_token = self.mlp_token(st_token)
            return logit, st_token


class ObjectStateChange(nn.Module):
    def __init__(self, num_frames, num_slots, encoder_dims, output_dim, attn_drop_t, path_drop, task):
        super(ObjectStateChange, self).__init__()
        self.T = num_frames
        self.num_slots = num_slots
        self.encoder_dims = encoder_dims
        self.task = task
        self.pos_drop = nn.Dropout(path_drop)
        self.st_transformer = SlotBlock(dim=self.encoder_dims, num_heads=8, drop=path_drop, attn_drop=attn_drop_t)

        if task == 'object':
            self.mlp_token = nn.Sequential(
                nn.Linear(self.encoder_dims, 2*self.encoder_dims), ##TODO: test mlp_ratio
                nn.ReLU(inplace=True),
                nn.Linear(2*self.encoder_dims, output_dim)
            )
        elif task == 'action':
            self.cls_layer = nn.Linear(self.encoder_dims, 174)
        elif task == 'joint':
            self.mlp_token = nn.Sequential(
                nn.Linear(2*self.encoder_dims, 2*self.encoder_dims), ##TODO: test mlp_ratio
                nn.ReLU(inplace=True),
                nn.Linear(2*self.encoder_dims, output_dim)
            )
            self.mlp_state = nn.Sequential(
                nn.Linear(output_dim, output_dim), ##TODO: test mlp_ratio
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, output_dim)
            )
            self.cls_layer = nn.Linear(output_dim, 174)

    def forward(self, slots_input):
        slots = self.st_transformer(slots_input)
        slots = einops.rearrange(slots, '(b t) s c -> b t s c', t=self.T)
        states_changes = []
        for i in range(1, self.T):
            states_change = torch.cat([slots[:, :-i], slots[:, i:]], dim=-1)
            states_changes.append(states_change)
        states_change = torch.cat(states_changes, dim=1)
        states_change = self.mlp_token(states_change.contiguous())
        states_all = states_change.max(dim=1)[0].mean(dim=1)
        logit = self.cls_layer(states_all)
        states_change = self.mlp_state(states_change)
        return logit, states_change


class SlotAttentionAutoEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""
    def __init__(self, resolution, 
                       num_slots, 
                       num_o=3, 
                       num_t=3,
                       in_channels=3, 
                       out_channels=3, 
                       hid_dim=32,
                       iters=5, 
                       path_drop=0.1,
                       attn_drop_t=0.2,
                       attn_drop_f=0.2,
                       num_frames=7,
                       background=False,
                       teacher=False,
                       slot_tune=False,
                       task='object',
                       student_model='dino',
                       teacher_model='dino',
                       correlation='none',
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
        self.num_slots = num_slots + 1 if background else num_slots
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T = num_frames
        
        self.student = student_model
        if self.student == 'dino':
            self.encoder_dims = 384
            self.encoder = vit_small(16, slot_tune=slot_tune, background=background, num_slots=num_slots)
            self.encoder.load_state_dict(torch.load(dino_path), strict=False)
            self.down_time = 16
        elif self.student == 'dinov2':
            self.encoder_dims = 384
            # self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.encoder = vit_small(14, block_chunks=0)
            self.encoder.load_state_dict(torch.load('dinov2_small_14.pth'))
            self.down_time = 14
        elif self.student == 'clip':
            self.encoder_dims = 768
            self.encoder, _ = clip.load('ViT-B/16')
            self.down_time = 16
        self.end_size = (resolution[0] // self.down_time, resolution[1] // self.down_time)
        
        self.num_t = num_t
        self.teacher = teacher
        self.correlation = correlation
        self.slot_tune = slot_tune
        self.task = task
        self.temporal_transformer = nn.ModuleList([Block(
                                                        dim=self.encoder_dims, 
                                                        num_heads=8,
                                                        n_token=num_frames, 
                                                        drop=path_drop,
                                                        attn_drop=attn_drop_t,
                                                        window=False,
                                                        num_frames=self.T,
                                                        end_size=self.end_size)
                                                        for j in range(self.num_t)])
        output_dim = self.encoder_dims if teacher_model == 'dino' else 512
        self.mlp_slot = nn.Sequential(
            nn.Linear(self.encoder_dims, 2*self.encoder_dims), ##TODO: test mlp_ratio
            nn.ReLU(inplace=True),
            nn.Linear(2*self.encoder_dims, output_dim)
        )
        self.slot_attention = SlotAttention(
            iters=self.iters,
            num_slots=self.num_slots,
            encoder_dims=self.encoder_dims,
            hidden_dim=hid_dim,
            background=background)
        
        if self.correlation == 'global':
            self.transport_proj = nn.Linear(self.end_size[0]*self.end_size[1], self.encoder_dims)
        elif self.correlation == 'cats':
            self.position_embed_x = nn.Parameter(
                torch.zeros(1, self.end_size[0], 1, (self.end_size[0]*self.end_size[1]+self.encoder_dims)//2))
            self.position_embed_y = nn.Parameter(
                torch.zeros(1, 1, self.end_size[1], (self.end_size[0]*self.end_size[1]+self.encoder_dims)//2))
            self.transport_proj = nn.Linear(self.end_size[0]*self.end_size[1]+self.encoder_dims, self.encoder_dims)
            self.attention = Attention(self.encoder_dims, attn_drop=attn_drop_t, proj_drop=path_drop)
        elif self.correlation == 'pos':
            self.position_embed = LinearPositionEmbedding(dim=128)
            self.transport_proj = nn.Linear(self.end_size[0]*self.end_size[1]+128, self.encoder_dims)
        elif self.correlation == 'window':
            self.window_sz = 5
            self.unfold = nn.Unfold(kernel_size=(self.window_sz, self.window_sz), padding=self.window_sz//2)
            self.position_embed = LinearPositionEmbedding(dim=128)
            self.transport_proj = nn.Linear(self.window_sz**2+128, self.encoder_dims)
        elif self.correlation == 'flow':
            self.patch_proj = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=6, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 64, kernel_size=6, stride=2, padding=2)
            )
            self.position_embed = LinearPositionEmbedding(dim=64)
            self.transport_proj = nn.Linear(self.end_size[0]//4*self.end_size[1]//4*64, self.encoder_dims)
        elif self.correlation == 'adapt':
            self.adapter = nn.Sequential(
                nn.Conv3d(self.encoder_dims, self.encoder_dims//4, kernel_size=1),
                nn.ReLU(),
                nn.Conv3d(self.encoder_dims//4, self.encoder_dims//4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(self.encoder_dims//4, self.encoder_dims, kernel_size=1)
            )
        
        self.st_block = ObjectStateChange(num_frames=num_frames, num_slots=num_slots, encoder_dims=self.encoder_dims, 
                                           output_dim=output_dim, attn_drop_t=attn_drop_t, path_drop=path_drop, task=task)

    def forward(self, image, weight=0.01, p_s=None):
        # Convolutional encoder with position embedding.
        bs = image.shape[0]
        image_t = einops.rearrange(image, 'b t c h w -> (b t) c h w')
        if self.student == 'dino':
            x, cls_token, attn, k = self.encoder(image_t)  # CNN Backbone/ DINO backbone
            x = einops.rearrange(x, '(b t) c h w -> b (t h w) c', t=self.T) ##spatial_flatten
            k = einops.rearrange(k, '(b t) c h w -> b (t h w) c', t=self.T)
            if self.num_t > 0:
                for block in self.temporal_transformer:
                    k = block(k)

            x = einops.rearrange(x, 'b (t hw) c -> b t hw c', t=self.T) ##spatial-temporal_map
            k = einops.rearrange(k, 'b (t hw) c -> b t hw c', t=self.T)
            if self.teacher:
                cls_token = einops.rearrange(cls_token, '(b t) c -> b t c', t=self.T)
                return x, cls_token

            init_slot = attn if self.slot_tune else None
            slots, motion_mask = self.decode(x, k, weight, ts=self.T, init_slot=init_slot)
        elif self.student == 'dinov2':
            x = self.encoder(image_t, is_training=True)
            k = x['k'][:, :, 1:].contiguous()
            k = einops.rearrange(k, '(b t) k hw c -> b t hw (k c)', t=self.T)
            x = x['x_norm_patchtokens']
            x = einops.rearrange(x, '(b t) hw c -> b t hw c', t=self.T)
            slots, motion_mask = self.decode(x, k, weight, ts=self.T)
        elif self.student == 'clip':
            cls_token, x = self.encoder.encode_image(image_t)
            cls_token = cls_token.float()
            x = x.float()
            x = einops.rearrange(x, '(b t) c h w -> b t (h w) c', t=self.T)
            slots, motion_mask = self.decode(x, x, weight, ts=self.T)
        
        st_token = self.st_block(slots)

        if p_s is None:
            motion_mask = F.softmax(motion_mask, dim=-2) + 1e-8
            return x, motion_mask, self.end_size[0]
        else:
            slots = self.mlp_slot(slots)
            slots = F.normalize(slots, dim=-1, p=2)
            return x, motion_mask, slots, st_token
    
    def decode(self, x, k, weight, ts=7, init_slot=None):
        bs = x.shape[0]
        if self.correlation == 'global':
            k_roll = torch.roll(k, shifts=1, dims=(1,))
            if self.end_size[0] * self.end_size[1] != k_roll.shape[-2]:
                k_roll = einops.rearrange(k_roll, 'b t (h w) c -> (b t) c h w', h=self.end_size[0])
                k_roll = F.interpolate(k_roll, (self.end_size[0], self.end_size[1]), mode='bilinear')
                k_roll = einops.rearrange(k_roll, '(b t) c h w -> b t (h w) c', t=self.T)
            correlation = torch.einsum('btnc,btmc->btnm', k, k_roll)
            correlation = einops.rearrange(correlation, 'b t n m -> (b t) n m')
            correlation = self.transport_proj(correlation)
            k = einops.rearrange(k, 'b t hw c -> (b t) hw c')
            slots, motion_mask = self.slot_attention(correlation+k, bs, weight, init_slot)
        elif self.correlation == 'cats':
            k_roll = torch.roll(k, shifts=1, dims=(1,))
            k_roll[:, 0] = k[:, 0]
            correlation = torch.einsum('btnc,btmc->btnm', k, k_roll)
            correlation = einops.rearrange(correlation, 'b t n m -> (b t) n m')
            k = einops.rearrange(k, 'b t hw c -> (b t) hw c')
            feat = torch.cat([correlation, k], dim=-1)
            position = torch.cat([self.position_embed_x.repeat(1, 1, self.end_size[1], 1), self.position_embed_y.repeat(1, self.end_size[0], 1, 1)], dim=-1)
            position = einops.rearrange(position, 'b h w c -> b (h w) c')
            feat = self.transport_proj(feat+position)
            feat = self.attention(feat) + feat
            slots, motion_mask = self.slot_attention(feat, bs, weight, init_slot)
        elif self.correlation == 'pos':
            k_roll = torch.roll(k, shifts=1, dims=(1,))
            correlation = torch.einsum('btnc,btmc->btnm', k, k_roll)
            correlation = einops.rearrange(correlation, 'b t (h w) m -> (b t) h w m', h=self.end_size[0])
            correlation = self.position_embed(correlation)
            correlation = einops.rearrange(correlation, 'b h w d -> b (h w) d')
            correlation = self.transport_proj(correlation)
            k = einops.rearrange(k, 'b t hw c -> (b t) hw c')
            slots, motion_mask = self.slot_attention(correlation+k, bs, weight, init_slot)
        elif self.correlation == 'window':
            k_roll = torch.roll(k, shifts=1, dims=(1,))
            k_roll[:, 0] = k[:, 0]
            k_roll = einops.rearrange(k_roll, 'b t (h w) c -> (b t) c h w', h=self.end_size[0])
            k_roll = self.unfold(k_roll).contiguous()
            k_roll = einops.rearrange(k_roll, '(b t) (ks c) hw -> b t ks c hw', t=self.T, ks=self.window_sz**2)
            correlation = torch.einsum('btnc,btkcn->btnk', k, k_roll)
            correlation = einops.rearrange(correlation, 'b t (h w) k -> (b t) h w k', h=self.end_size[0])
            correlation = self.position_embed(correlation)
            correlation = einops.rearrange(correlation, 'b h w d -> b (h w) d')
            correlation = self.transport_proj(correlation)
            k = einops.rearrange(k, 'b t hw c -> (b t) hw c')
            slots, motion_mask = self.slot_attention(correlation+k, bs, weight, init_slot)
        elif self.correlation == 'flow':
            k_roll = torch.roll(k, shifts=1, dims=(1,))
            k_roll[:, 0] = k[:, 0]
            correlation = torch.einsum('btnc,btmc->btnm', k, k_roll)
            correlation = einops.rearrange(correlation, 'b t n (h w) -> (b t n) h w', h=self.end_size[0])
            correlation = self.patch_proj(correlation.unsqueeze(1))
            correlation = einops.rearrange(correlation, 'b c h w -> b h w c')
            correlation = self.position_embed(correlation, fusion='sum')
            correlation = einops.rearrange(correlation, '(b n) h w c -> b n (h w c)', n=self.end_size[0]*self.end_size[1])
            correlation = self.transport_proj(correlation)
            k = einops.rearrange(k, 'b t hw c -> (b t) hw c')
            slots, motion_mask = self.slot_attention(correlation+k, bs, weight, init_slot)
        elif self.correlation == 'adapt':
            k = einops.rearrange(k, 'b t (h w) c -> b c t h w', h=self.end_size[0])
            k = k + self.adapter(k)
            k = einops.rearrange(k, 'b c t h w -> (b t) (h w) c')
            slots, motion_mask = self.slot_attention(k, bs, weight, init_slot)
        else:
            x = einops.rearrange(x, 'b t hw c -> (b t) hw c')
            k = einops.rearrange(k, 'b t hw c -> (b t) hw c')
            slots, motion_mask = self.slot_attention(k, bs, weight, init_slot) 
        slots = einops.rearrange(slots, '(b t) s c -> b t s c', b=bs)
        motion_mask = einops.rearrange(motion_mask, '(b t) s hw -> b t s hw', b=bs)
        return slots, motion_mask


if __name__ == "__main__":
    model = SlotAttentionAutoEncoder(resolution=(192, 384), 
                       num_slots=2, 
                       num_o=3, 
                       num_t=3,
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
