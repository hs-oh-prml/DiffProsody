import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SingleWindowDisc(nn.Module):
    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.mlp1 = nn.Linear(freq_length, hidden_size, 1)
        self.mlp2 = nn.Linear(192, hidden_size, 1)

        self.model = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(c_in, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),            
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]),
        ])
        
        self.model2 = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(c_in, hidden_size, 3, (2, 2), 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),            
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ])
        ])
        ds_size = (time_length // 2 ** 3, (freq_length + 7) // 2 ** 3)
        ds_size2 = (time_length // 2 ** 3, (384 + 7) // 2 ** 3)
        self.adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1] + hidden_size * ds_size2[0] * ds_size2[1], 1)

    def forward(self, x, cond=None):
        """
        :param x: [B, C, T, n_bins]
        :return: validity: [B, 1], h: List of hiddens
        """
        h = []
        if cond is not None:
            x_ = self.mlp1(x)
            cond = self.mlp2(cond)

        for idx, l in enumerate(self.model):           
            x = l(x)
            h.append(x)

        cond = torch.cat([cond, x_], dim=-1)
        for idx, l in enumerate(self.model2):
            cond = l(cond)
            h.append(cond)
        x = x.view(x.shape[0], -1)
        cond = cond.view(cond.shape[0], -1)
        x = torch.cat([x, cond], dim=1)
        validity = self.adv_layer(x)  # [B, 1]

        return validity, h

class MultiWindowDiscriminator(nn.Module):
    def __init__(self, time_lengths, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.discriminators = nn.ModuleList()

        for time_length in time_lengths:
            self.discriminators += [SingleWindowDisc(time_length, freq_length, kernel, c_in=c_in, hidden_size=hidden_size)]
    
    def forward(self, x, x_len, cond=None, start_frames_wins=None):
        '''
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).
        Returns:
            tensor : (B).
        '''
        validity = []

        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.discriminators)
        h = []
        for i, start_frames in zip(range(len(self.discriminators)), start_frames_wins):
            x_clip, start_frames, cond_clip = self.clip(x, x_len, self.win_lengths[i], start_frames, cond=cond)  # (B, win_length, C)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            x_clip, h_ = self.discriminators[i](x_clip, cond_clip)
            h += h_
            validity.append(x_clip)
        validity = sum(validity)  # [B]

        return validity, start_frames_wins, h

    def clip(self, x, x_len, win_length, start_frames=None, cond=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length
        Returns:
            (tensor) : (B, c_in, win_length, n_bins).
        '''
        T_start = 0
        T_end = x_len.max() - win_length
        if T_end < 0:
            return None, start_frames, None

        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]
        x_batch = x[:, :, start_frame: start_frame + win_length]

        if cond is not None:
            cond_batch = cond[:, :, start_frame: start_frame + win_length]
            
        return x_batch, start_frames, cond_batch

class Discriminator(nn.Module):
    def __init__(self, time_lengths=[32, 64, 128], 
                    freq_length=80, kernel=(3, 3), c_in=1,
                    hidden_size=128):
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.discriminator = MultiWindowDiscriminator(
            freq_length=freq_length,
            time_lengths=time_lengths,
            kernel=kernel,
            c_in=c_in, hidden_size=hidden_size
        )
        
    def forward(self, x, cond=None, word_mask=None, start_frames_wins=None):
        """
        :param x: [B, T, 80]
        :param return_y_only:
        :return:
        """
        if len(x.shape) == 3:
            x = x[:, None, :, :] # [B,1,T,80]
            cond = cond[:, None, :, :] # [B,1,T,192]
        x_len = x.sum([1, -1]).ne(0).int().sum([-1])

        ret = {'y': None}
        ret['y'], start_frames_wins, ret['h'] = self.discriminator(
            x, x_len, cond=cond, start_frames_wins=start_frames_wins)
        
        ret['start_frames_wins'] = start_frames_wins
        return ret

class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x

class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal

class DiffusionEmbedding(nn.Module):
    """ Diffusion Step Embedding """

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffDiscriminator(nn.Module):
    """ JCU Discriminator """

    def __init__(self, hparams, in_dim=192):
        super(DiffDiscriminator, self).__init__()

        n_mel_channels = in_dim
        residual_channels = hparams["disc_hidden_size"]
        n_layer = hparams["n_layer"]
        n_channels = hparams["disc_n_channels"]
        kernel_sizes = hparams["disc_kernel_sizes"]
        strides = hparams["disc_strides"]

        self.input_projection = LinearNorm(2 * n_mel_channels, n_channels[0])
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, n_channels[0]),
        )
        self.pre = nn.Sequential(
            ConvNorm(n_mel_channels, n_channels[0]),
        )
        self.conv_block = nn.ModuleList(
            [
                ConvNorm(
                        n_channels[i-1],
                        n_channels[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        dilation=1,
                    )
                    for i in range(1, n_layer)
            ]
        )
        self.apply(self.weights_init)
        self.adv_layer = ConvNorm(n_channels[-1], 1)
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("ConvNorm") != -1:
            m.conv.weight.data.normal_(0.0, 0.02)

    def forward(self, x, t, x_t, s, mask=None):        # x_ts, x_t_prevs, s, t
        """
        x_ts -- [B, T, H]
        x_t_prevs -- [B, T, H]
        s -- [B, H]
        t -- [B]
        """
        x = self.input_projection(torch.cat([x, x_t], dim=-1)).transpose(1, 2)
        diffusion_step = self.mlp(self.diffusion_embedding(t)).unsqueeze(-1)
        cond = self.pre(s)
        cond_feats = []
        x = (x + diffusion_step + cond) 
        for layer in self.conv_block:
            x = F.leaky_relu(layer(x), 0.2)
            cond_feats.append(x)
        x = self.adv_layer(x)
        cond_feats.append(x)
        return cond_feats
        