import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from modules.commons.layers import Embedding
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.tts.commons.align_ops import build_word_mask, expand_states
from modules.tts.fs import FS_DECODERS, FastSpeech
from .prosody_encoder import ProsodyEncoder

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        :param x: [B, T]
        :return: [B, T, H]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, :, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class DiffProsody(FastSpeech):
    def __init__(self, 
                ph_dict_size, word_dict_size, 
                hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        # build linguistic encoder
        self.word_encoder = RelTransformerEncoder(
            word_dict_size, self.hidden_size, self.hidden_size, self.hidden_size, 2,
            hparams['word_enc_layers'], hparams['enc_ffn_kernel_size'])

        self.sin_pos = SinusoidalPosEmb(self.hidden_size)
 
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.prosody_encoder = ProsodyEncoder(hparams)

        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        self.word_pos_proj = Linear(self.hidden_size, self.hidden_size)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, word_tokens, ph2word, word_len, mel2word=None, mel2ph=None,
                spk_embed=None, spk_id=None, pitch=None, infer=False, tgt_mels=None, bert_tokens=None,
                global_step=None, lpv=None, *args, **kwargs):
        ret = {}
        # print(bert_tokens.shape, word_tokens.shape)
        h_ling = self.run_text_encoder(
            txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, ret)
        h_spk = self.forward_style_embed(spk_embed, spk_id)

        vq_loss = None
        if not infer:
            wrd_nonpadding = (word_tokens > 0).float()[:, :, None]        
            vq_loss, lpv, lpv_idx, perplexity, _ = self.prosody_encoder(tgt_mels, h_ling, h_spk, 
                                            mel2word, mel2ph, ph2word, word_len, 
                                            wrd_nonpadding, global_step)
            if global_step > self.hparams["vq_warmup"]:
                # print(wrd_nonpadding.shape, lpv_idx.shape)
                lpv_idx = lpv_idx.masked_select(wrd_nonpadding.unsqueeze(-1).bool())
                ret['lpv_idx'] = lpv_idx
                ret['perplexity'] = perplexity
        else:
            assert lpv is not None, 'LPV required for inference'
        x = h_ling + h_spk + expand_states(lpv, ph2word)
        mel2ph = self.forward_dur(x, mel2ph, txt_tokens, ret)
        # mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        
        x = expand_states(x, mel2ph)
        x = x * tgt_nonpadding
        
        ret['nonpadding'] = tgt_nonpadding
        ret['decoder_inp'] = x
        ret['lpv'] = lpv
        ret['lpv_long'] = expand_states(expand_states(lpv, ph2word), mel2ph)
        
        ret['vq_loss'] = vq_loss
        ret['mel_out'] = self.run_decoder(x, tgt_nonpadding, ret, infer, tgt_mels, global_step)

        return ret

    def forward_style_embed(self, spk_embed=None, spk_id=None):
        # add spk embed
        # style_embed = self.spk_id_proj(spk_id)[:, None, :]
        style_embed = self.spk_embed_proj(spk_embed)[:, None, :]
        return style_embed

    def get_lpv(self, 
                txt_tokens, word_tokens, 
                ph2word, word_len, 
                mel2word, mel2ph, 
                spk_embed,
                tgt_mels, global_step):
        ret = {}
        h_ling = self.run_text_encoder(
            txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, ret)
        h_spk = self.forward_style_embed(spk_embed)

        wrd_nonpadding = (word_tokens > 0).float()[:, :, None]
        _, _, idx, _, lpv = self.prosody_encoder(tgt_mels, h_ling, h_spk, 
                                        mel2word, mel2ph, ph2word, word_len, 
                                        wrd_nonpadding, global_step)
        return lpv, idx

    def run_text_encoder(self, txt_tokens, word_tokens, ph2word, word_len, mel2word, mel2ph, ret):
        src_nonpadding = (txt_tokens > 0).float()[:, :, None] 
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
        word_encoder_out = self.word_encoder(word_tokens)
        ph_encoder_out = ph_encoder_out + expand_states(word_encoder_out, ph2word)

        return ph_encoder_out

    def run_decoder(self, x, tgt_nonpadding, ret, infer, tgt_mels=None, global_step=0):
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding

    def get_pos_embed(self, word2word, x2word):
        x_pos = build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos
    
    @property
    def device(self):
        return next(self.parameters()).device