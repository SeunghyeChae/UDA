import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, hidden_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    # Multihead attention
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)  

        self.attend = nn.Softmax(dim = -1)  
        self.dropout = nn.Dropout(dropout)  

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # Q, K, V 생성을 위한 Linear transform

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  
            nn.Dropout(dropout)  
        ) if project_out else nn.Identity()  # 출력 projection 적용 여부

    def forward(self, x):
        x = self.norm(x) 

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # Q, K, V 분할
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # Q, K, V 재배열

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # scaled dot product

        attn = self.attend(dots)  # attention map 
        attn = self.dropout(attn)  

        out = torch.matmul(attn, v)  # attention 적용
        out = rearrange(out, 'b h n d -> b n (h d)')  # output 재배열 
        return self.to_out(out)  # 최종 output

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),  # attention layer
                FeedForward(dim, mlp_dim, dropout = dropout)  # FF layer
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # attention 적용 및 residual connection
            x = ff(x) + x  # Feedforward 적용 및 residual connection

        return self.norm(x)  

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)  
        patch_height, patch_width = pair(patch_size)  

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  
        patch_dim = channels * patch_height * patch_width  
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),  # patch embedding 
            nn.LayerNorm(patch_dim),  
            nn.Linear(patch_dim, dim), 
            nn.LayerNorm(dim),  
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # positional embedding/ class token이랑 결합해야해서 num_patches+1
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # class token
        self.dropout = nn.Dropout(emb_dropout) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) 

        self.pool = pool
        self.to_latent = nn.Identity() 

        self.mlp_head = nn.Linear(dim, num_classes)  # 최종분류 mlp

    def forward(self, img):
        x = self.to_patch_embedding(img)  # patch embedding
        b, n, _ = x.shape  # batch_size , num_patch

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  # class token 복제 / 왜함? 
        x = torch.cat((cls_tokens, x), dim=1)  # class token 과 patch embedding 결합
        x += self.pos_embedding[:, :(n + 1)]  # positional embedding 추가
        x = self.dropout(x)  

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x) 
        return self.mlp_head(x)  
