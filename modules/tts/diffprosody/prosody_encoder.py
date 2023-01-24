import torch
import torch.nn as nn 
import torch.nn.functional as F
from modules.commons.conv import ConditionalConvBlocks, ConvBlocks
from utils.nn.seq_utils import group_hidden_by_segs
from modules.tts.commons.align_ops import expand_states

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, 
                cluster_centers=None, decay=0.996, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        if cluster_centers is not None:
            self._embedding = nn.Embedding.from_pretrained(cluster_centers)
        else:
            self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self._embedding.weight.data.uniform_(-1.0 / self._num_embeddings, 1.0 / self._embedding_dim)

        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten        
        quantized = self._embedding(encoding_indices).view(inputs.shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        encoding_indices = encoding_indices.view(input_shape[0], input_shape[1])
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings, encoding_indices

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

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

class ProsodyEncoder(nn.Module):
    def __init__(self, hparams, ln_eps=1e-12):
        super(ProsodyEncoder, self).__init__()
        self.hparams = hparams
        self.n = 5
        self.hidden_size = self.hparams["hidden_size"]
        self.kernel_size = 5
        self.n_embeddings = 128
        self.embedding_dim = self.hparams["hidden_size"]
        self.beta = self.hparams["commitment_cost"]
        print("Prosody mel bins: ", self.hparams["prosody_mel_bins"])
        self.pre = nn.Sequential(
            nn.Linear(self.hparams["prosody_mel_bins"], self.hidden_size // 4),
            Mish(),
            nn.Linear(self.hidden_size // 4, self.hidden_size)
        )
        self.conv1 = ConditionalConvBlocks(self.hidden_size, self.hidden_size, self.hidden_size,
                                            None, self.kernel_size, num_layers=self.n)
        self.conv2 = ConditionalConvBlocks(self.hidden_size, self.hidden_size, self.hidden_size,
                                            None, self.kernel_size, num_layers=self.n)
        self.vector_quantization = None
        self.post_net = ConvBlocks(self.hidden_size, self.hidden_size, None, 1, num_layers=3)

    def init_vq(self, cluster_centers):
        self.vector_quantization = VectorQuantizerEMA(
            self.n_embeddings, self.embedding_dim, self.beta, cluster_centers=cluster_centers, decay=self.hparams["ema_decay"]).cuda()
        print("Initialized Codebook with cluster centers [EMA]")
    def forward(self, x, h_lin, h_spk, mel2word, mel2ph, ph2word, word_len, wrd_nonpadding, global_step=None):
        x = x[:, :, :self.hparams["prosody_mel_bins"]]      # Mel: 80 bin -> 20 bin
        x = self.pre(x)

        cond = h_lin + h_spk # Phoneme-level
        cond1 = expand_states(cond, mel2ph) # Frame level
        x = self.conv1(x, cond1)
        x = group_hidden_by_segs(x, mel2word, word_len)[0]  # Word-level 
        cond2 = group_hidden_by_segs(cond, ph2word, word_len)[0]
        x = self.conv2(x, cond2)
        x = self.post_net(x)

        if global_step > self.hparams["vq_warmup"]:
            embedding_loss, x2, perplexity, min_encodings, min_encoding_indices = self.vector_quantization(x) # VQ
            x2 = x2 * wrd_nonpadding
            return embedding_loss, x2, min_encoding_indices, perplexity, x
        else:
            return None, x, None, None, x