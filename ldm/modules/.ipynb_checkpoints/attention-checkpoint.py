from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from ldm.modules.diffusionmodules.util import checkpoint
# from new_module.module052 import FFCM
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
# --------------------------------------------------------------
class ReLUX(nn.Module):
    def __init__(self, thre=8):  # thre是输出的上限值
        super(ReLUX, self).__init__()
        self.thre = thre  # 保存上限值

    def forward(self, input):
        # 使用torch.clamp将输入限制在[0, thre]之间
        return torch.clamp(input, 0, self.thre)

# 定义一个ReLUX实例，thre=4
relu4 = ReLUX(thre=4)
class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens):  # 前向传播
        ctx.save_for_backward(input)  # 保存输入以备反向传播使用
        ctx.lens = lens  # 保存脉冲长度
        return torch.floor(relu4(input) + 0.5)  # 应用ReLUX并进行四舍五入

    @staticmethod
    def backward(ctx, grad_output):  # 反向传播
        input, = ctx.saved_tensors  # 获取保存的输入
        grad_input = grad_output.clone()  # 克隆梯度输出
        temp1 = 0 < input  # 判断输入是否大于0
        temp2 = input < ctx.lens  # 判断输入是否小于脉冲长度
        return grad_input * temp1.float() * temp2.float(), None  # 返回梯度

# 定义一个多脉冲激活模块，封装multispike
class Multispike(nn.Module):
    def __init__(self, lens=4):  # lens是脉冲长度
        super().__init__()
        self.lens = lens  # 保存脉冲长度
        self.spike = multispike  # 保存激活函数

    def forward(self, inputs):
        # 调用激活函数并归一化
        return self.spike.apply(4 * inputs, self.lens) / 4

# 定义一个多脉冲注意力模块，与Multispike类似，但归一化因子不同
class Multispike_att(nn.Module):
    def __init__(self, lens=4):
        super().__init__()
        self.lens = lens
        self.spike = multispike

    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 2
#----------------------------------------------------------------------


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        # total_params = sum(p.numel() for p in self.net.parameters())
        # print(f'Total parameters: {total_params / 1e6:.2f}M')
        # Total parameters: 19.67M
        # Total parameters: 4.92M
        # Total parameters: 4.92M
        # Total parameters: 1.23M
        # Total parameters: 1.23M 还是有点大

        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

# 还是该这里
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads

        self.scale = dim_head ** -0.5
        # self.att_net = TopK_Sparse_Attention(dim=inner_dim, num_heads=heads, bias=False).to(device='cuda:0').half()
        # self.m123 = ASSA(dim=inner_dim, input_resolution=(16, 16), num_heads=heads).to(device='cuda:0').half()
        # self.m129 = AttentionTSSA(dim=inner_dim, num_heads=heads).to(device='cuda:0').half()
        # self.m052 = FFCM(dim=inner_dim).to(device='cuda:0').half()
        self.dim_head = dim_head
        # 如果 val 存在（即 exists(val) 为 True），则返回 val；否则返回默认值 d。默认值 d 可以是一个直接的值，
        # 也可以是一个函数，如果是函数，则调用 d() 并返回其结果。
        # 自定义的函数，实现在上面
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5


        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        # -------------------------------------------------------
        # self.q_lif = Multispike()
        # self.k_lif = Multispike()
        # self.v_lif = Multispike()
        # -------------------------------------------------------


    def forward(self, x, context=None, mask=None):
        # print('走得交叉注意力')
        b, hw, c = x.shape
        hh = int(math.sqrt(hw))
        context = default(context, x)
        # if context.shape == x.shape:# 完全相等时是自注意机制 # 换023
        #     # return self.m129(x)
        #     x = to_4d(x, hh, hh)
        #     # print('x shape:', x.shape)
        #     x = self.m052(x)
        #     return to_3d(x)
        # print('context shape:', context.shape) context shape: torch.Size([8, 77, 1024])

        h = self.heads
        # print(2222, h)
        # print('CrossAttention x shape:', x.shape) # CrossAttention x shape: torch.Size([4, 4096, 320])
        # 对k、q、v进行量化脉冲
        q = self.to_q(x)
        # ---------------------------------------
        # q = to_4d(q, hh, hh).unsqueeze(0)
        # q = self.q_lif(q).flatten(3).squeeze(0).permute(0, 2, 1)
        # ----------------------------------------
        # print('000q shape:', q.shape)
        # q shape: torch.Size([4, 4096, 320])
        # k shape: torch.Size([4, 77, 320]) 77是n
        # context = default(context, x)
        k = self.to_k(context.half())
        # k = to_4d(k, 7, 11).unsqueeze(0)
        # k = self.k_lif(k).flatten(3).squeeze(0).permute(0, 2, 1)
        # print('000k shape:', k.shape)
        v = self.to_v(context.half())
        # v = to_4d(v, 7, 11).unsqueeze(0)
        # v = self.v_lif(v).flatten(3).squeeze(0).permute(0, 2, 1)
        # print('000v shape:', v.shape)

        # b 是批量大小（batch size）。
        # n 是序列长度（sequence length）。
        # h*d 是特征维度，其中 h 是头数（number of heads），d 是每个头的维度。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print('333q shape:', q.shape)
        # print('333k shape:', k.shape)
        # print('333v shape:', v.shape)
        # 333q shape: torch.Size([80, 1024, 64])
        # 333k shape: torch.Size([80, 77, 64])
        # 333v shape: torch.Size([80, 77, 64])

        # force cast to fp32 to avoid overflowing
        # einsum计算相似性矩阵
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        # print('sim shape:', sim.shape) # sim shape: torch.Size([10, 4096, 4096])  torch.Size([10, 4096, 77])
        out = einsum('b i j, b j d -> b i d', sim.to(torch.float16), v.to(torch.float16))
        # out = einsum('b i j, b j d -> b i d', sim.to(torch.float16), v.to(torch.float16))

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # print('out shape', out.shape)
        return self.to_out(out.half())


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

        # self.m052 = FFCM(dim=inner_dim).to(device='cuda:0').half()
        # -------------------------------------------------------
        self.q_lif = Multispike()
        self.k_lif = Multispike()
        self.v_lif = Multispike()
        # -------------------------------------------------------

    def forward(self, x, context=None, mask=None):
        b, hw, c = x.shape
        hh = int(math.sqrt(hw))
        context = default(context, x)
        # if context.shape != x.shape:# 完全相等时是自注意机制 # 换023
        #     h = self.heads
        #     q = self.to_q(x)
        #     q = to_4d(q, hh, hh).unsqueeze(0)
        #     q = self.q_lif(q).flatten(3).squeeze(0).permute(0, 2, 1)
        #     k = self.to_k(context)
        #     k = to_4d(k, 7, 11).unsqueeze(0)
        #     k = self.k_lif(k).flatten(3).squeeze(0).permute(0, 2, 1)
        #     v = self.to_v(context)
        #     v = to_4d(v, 7, 11).unsqueeze(0)
        #     v = self.v_lif(v).flatten(3).squeeze(0).permute(0, 2, 1)
        # #     x = to_4d(x, hh, hh)
        # #     x = self.m052(x)
        # #     return to_3d(x)
            
        # else:
        #     h = self.heads
        #     q = self.to_q(x)
        #     q = to_4d(q, hh, hh).unsqueeze(0)
        #     q = self.q_lif(q).flatten(3).squeeze(0).permute(0, 2, 1)
        #     k = self.to_k(context)
        #     k = to_4d(k, hh, hh).unsqueeze(0)
        #     k = self.k_lif(k).flatten(3).squeeze(0).permute(0, 2, 1)
        #     v = self.to_v(context)
        #     v = to_4d(v, hh, hh).unsqueeze(0)
        #     v = self.v_lif(v).flatten(3).squeeze(0).permute(0, 2, 1)
        
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

# 改这里 transformer块------------------------------------------------
class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # self.m124 = FRFN(dim=dim)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        
    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

# transformer主要使用这个块
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        # print('depth:', depth)是1，因为只用接1个
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

