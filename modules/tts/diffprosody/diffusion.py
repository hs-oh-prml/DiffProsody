import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from functools import partial
from inspect import isfunction
import numpy as np
import json 
import os

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)

class ContextEncoder(nn.Module):

    def __init__(self, vocab_size, hidden=384, n_layers=6, attn_heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = WordEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        # print(scores.shape, mask.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
#################################################


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        in_dims = hparams['hidden_size']
        self.encoder_hidden = hparams['hidden_size']
        self.residual_layers = hparams['residual_layers']
        self.residual_channels = hparams['residual_channels']
        self.dilation_cycle_length = hparams['dilation_cycle_length']

        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.encoder_hidden, self.residual_channels, 2 ** (i % self.dilation_cycle_length))
            for i in range(self.residual_layers)
        ])
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        return x[:, None, :, :]

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - np.exp(2. * log_mean_coeff)
    return var

def sigma_beta_schedule(timesteps, min_beta=0.01, max_beta=20, use_geometric=False):
    eps_small = 1e-3
   
    t = np.arange(0, timesteps + 1, dtype=np.float64)
    t = t / timesteps
    t = t * (1. - eps_small) + eps_small
    
    if use_geometric:
        var = var_func_geometric(t, min_beta, max_beta)
    else:
        var = var_func_vp(t, min_beta, max_beta)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = np.array([1e-8])
    betas = np.concatenate((first, betas))
    sigmas = betas**0.5
    a_s = np.sqrt(1-betas)
    return sigmas, a_s, betas

def linear_beta_schedule(timesteps, max_beta=0.01):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp),
}

class DiffusionProsodyGenerator(nn.Module):
    def __init__(self, hparams, out_dims=None):
        super().__init__()
        self.hparams = hparams
        out_dims = hparams['hidden_size']
        denoise_fn = DIFF_DECODERS[hparams['diff_decoder_type']](hparams)
        timesteps = hparams['timesteps']
        K_step = hparams['K_step']
        loss_type = hparams['diff_loss_type']

        stats_f = os.path.join(hparams['tts_model'], 
                "stats_lpv_{}.json".format(hparams['train_set_name']))
                    
        with open(stats_f) as f:
            stats = json.load(f)
        spec_min = stats['lpv_min'][0]
        spec_max = stats['lpv_max'][0]

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims

        sigmas, a_s, betas = sigma_beta_schedule(timesteps, hparams['min_beta'], hparams['max_beta'])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
        self.loss_type = loss_type
        self.proj = nn.Linear(384, 192)

        a_s_cum = np.cumprod(a_s)
        sigmas_cum = np.sqrt(1 - a_s_cum ** 2)
        a_s_prev = np.copy(a_s)
        a_s_prev[-1] = 1

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('a_s', to_torch(a_s))
        self.register_buffer('sigmas', to_torch(sigmas))

        self.register_buffer('a_s_prev', to_torch(a_s_prev))
        self.register_buffer('a_s_cum', to_torch(a_s_cum))
        self.register_buffer('sigmas_cum', to_torch(sigmas_cum))

        # Posterior coefficients
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas, 0)
        alphas_cumprod_prev = np.concatenate((np.array([1.]), alphas_cumprod[:-1]))
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('posterior_mean_coef1', to_torch((betas * np.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))))
        self.register_buffer('posterior_mean_coef2', to_torch((1 - alphas_cumprod_prev) * np.sqrt(alphas) / (1 - alphas_cumprod)))
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', torch.log(to_torch(posterior_variance).clamp(min=1e-20)))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(self, x_start, x_t, t, repeat_noise=False):
        b, *_, device = *x_start.shape, x_start.device
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = noise_like(x_start.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.a_s_cum, t, x_start.shape) * x_start +
                extract(self.sigmas_cum, t, x_start.shape) * noise
        )

    def q_sample_pairs(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_plus_one = extract(self.a_s, t+1, x_start.shape) * x_start + \
                extract(self.sigmas, t+1, x_start.shape) * noise

        return x_t, x_t_plus_one

    def forward(self, word_tokens, spk_embed=None, spk_id=None, lpv=None,
                ph2word=None, infer=False, padding=None, noise_scale=0.1, **kwargs):
        b, *_, device = *word_tokens.shape, word_tokens.device

        cond = (word_tokens + spk_embed.unsqueeze(1)).transpose(1, 2)

        ret = {}
        if padding is not None:
            padding = padding.unsqueeze(1).unsqueeze(1)

        if not infer:
            t = torch.randint(0, self.K_step, (b,), device=device).long()
            x = lpv
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            
            noise = default(None, lambda: torch.randn_like(x))

            x_t, x_tp1 = self.q_sample_pairs(x_start=x, t=t, noise=noise)
            x_0_predict = self.denoise_fn(x_tp1, t, cond)
            x_0_predict.clamp_(-1., 1.)
            x_pos_sample = self.q_posterior_sample(x_0_predict, x_tp1, t)
            
            x_0_predict = x_0_predict * padding
            x_t = x_t * padding
            x_tp1 = x_tp1 * padding
            x_pos_sample = x_pos_sample * padding

            x_0_predict = x_0_predict[:, 0].transpose(1, 2)
            x_t = x_t[:, 0].transpose(1, 2)
            x_tp1 = x_tp1[:, 0].transpose(1, 2)
            x_pos_sample = x_pos_sample[:, 0].transpose(1, 2)
            
            ret["x_0_predict"] = x_0_predict
            ret["x_t"] = x_t
            ret["x_tp1"] = x_tp1
            ret["x_pos_sample"] = x_pos_sample
            ret["t"] = t
            ret["cond"] = cond
            ret['lpv_out'] = None

        else:
            t = self.K_step
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])
            x = torch.randn(shape, device=device) * noise_scale

            for i in reversed(range(0, t)):
                t_i = torch.full((x.size(0),), i, device=device).long()
                x_0 = self.denoise_fn(x, t_i, cond=cond)
                x_new = self.q_posterior_sample(x_0, x, t_i)
                x = x_new.detach()

            x = x[:, 0].transpose(1, 2)
            ret['lpv_out'] = self.denorm_spec(x)
        return ret

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min