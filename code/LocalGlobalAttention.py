import torch
from torch import nn

class LocalGlobalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, local_kernel_sizes=[3, 5, 7], global_kernel_size=11, height=14, width=14,
                 num_scales=3):
        super(LocalGlobalAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_scales = num_scales

        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=2 ** i, groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            )
            for i in range(num_scales)
        ])

        self.local_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=k, padding=k // 2, groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1, groups=num_heads)
            )
            for k in local_kernel_sizes
        ])

        self.global_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=global_kernel_size, padding=global_kernel_size // 2,
                      groups=embed_dim),
            nn.Conv2d(embed_dim, embed_dim * 3, kernel_size=1, groups=num_heads)
        )

        self.position_encoding = nn.Parameter(torch.randn(1, embed_dim, height, width))

        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.scale_attention = nn.Sequential(
            nn.Conv2d(embed_dim, num_scales + 1, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.learnable_alpha_local = nn.Parameter(torch.tensor(0.5))
        self.learnable_alpha_global = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        batch_size, embed_dim, height, width = x.size()

        def split_heads(tensor):
            return tensor.view(batch_size, self.num_heads, self.head_dim, -1)

        def combine_heads(tensor):
            return tensor.view(batch_size, embed_dim, height, width)

        position_encoding = nn.functional.interpolate(self.position_encoding, size=(height, width), mode='nearest')

        multi_scale_features = [x]
        for scale_conv in self.multi_scale_convs:
            scaled_x = scale_conv(x)
            scaled_x = nn.functional.interpolate(scaled_x, size=(height, width), mode='nearest')
            scaled_x += x
            multi_scale_features.append(scaled_x)
        scale_weights = self.scale_attention(x)
        x_multi_scale = sum(scale_weights[:, i:i+1, :, :] * multi_scale_features[i]
                            for i in range(self.num_scales + 1))


        local_outs = []
        for local_conv in self.local_convs:
            local_query, local_key, local_value = local_conv(x_multi_scale + position_encoding).chunk(3, dim=1)
            local_query, local_key, local_value = split_heads(local_query), split_heads(local_key), split_heads(local_value)

            local_energy = torch.einsum('bhqd, bhkd -> bhqk', local_query, local_key) * self.scale
            local_attention = self.softmax(local_energy)
            local_out = torch.einsum('bhqk, bhvd -> bhqd', local_attention, local_value)
            local_out = combine_heads(local_out) + x
            local_outs.append(local_out)
        local_out = sum(local_outs) / len(local_outs)

        global_query, global_key, global_value = self.global_conv(x_multi_scale + position_encoding).chunk(3, dim=1)
        global_query, global_key, global_value = split_heads(global_query), split_heads(global_key), split_heads(global_value)
        global_energy = torch.einsum('bhqd, bhkd -> bhqk', global_query, global_key) * self.scale
        global_attention = self.softmax(global_energy)
        global_out = torch.einsum('bhqk, bhvd -> bhqd', global_attention, global_value)
        global_out = combine_heads(global_out) + x

        out = self.learnable_alpha_local * local_out + self.learnable_alpha_global * global_out
        out = self.out_conv(out)

        return out
