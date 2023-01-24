import os
import torch
import torch.nn.functional as F
from modules.tts.diffprosody.diffprosody import DiffProsody
from modules.tts.diffprosody.discriminator import Discriminator
from tasks.tts.fs import FastSpeechTask
from utils.audio.align import mel2token_to_dur
from utils.commons.hparams import hparams
from utils.nn.model_utils import num_params
import numpy as np
from utils.text.text_encoder import build_token_encoder
from utils.commons.tensor_utils import move_to_cuda
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

class DiffProsodyTask(FastSpeechTask):
    def __init__(self):
        super().__init__()
        data_dir = hparams['binary_data_dir']
        self.word_encoder = build_token_encoder(f'{data_dir}/word_set.json')
        self.is_lpv_init = False
        self.lpv_dist = None
        self.mse_loss_fn = torch.nn.MSELoss()
        
    def build_tts_model(self):
        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        self.model = DiffProsody(ph_dict_size, word_dict_size, hparams)
        if os.path.exists("{}/lpvs.npy".format(hparams["work_dir"])):
            self.model.prosody_encoder.init_vq(None)
        
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']            
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())
        
    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        for n, m in self.model.prosody_encoder.named_children():
            num_params(m, model_name=f'p_enc.{n}')
        for n, m in self.mel_disc.named_children():
            num_params(m, model_name=f'disc.{n}')

    def cluster_and_init(self):

        dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset,
                                    collate_fn=dataset.collater,
                                    batch_size=1,     # Numer of Sample 
                                    num_workers=8,
                                    pin_memory=False)
        print("###### Cluster Dataloader: ", len(dataloader))
        self.model.eval()
        if os.path.exists("{}/lpvs.npy".format(hparams["work_dir"])):
            lpvs = np.load("{}/lpvs.npy".format(hparams["work_dir"]))
        else:
            lpvs = None 
            word_lbl = None
            spk_lbl = None
            for idx, i in tqdm(enumerate(dataloader)):
                i = move_to_cuda(i, self.model.device)
                word = i["word_tokens"]
                lpv, _ = self.model.get_lpv(
                    i["txt_tokens"], word,
                    i["ph2word"], i["word_lengths"].max(),
                    i["mel2word"], i["mel2ph"],
                    i.get('spk_embed'),
                    i['mels'], self.global_step
                )
                if lpvs is None:
                    lpvs = lpv.flatten(0, 1)    
                else:
                    lpvs = torch.cat([lpvs, lpv.flatten(0, 1)], dim=0)
                if word_lbl is None:
                    word_lbl = word.flatten(0, 1)
                else:
                    word_lbl = torch.cat([word_lbl, word.flatten(0, 1)], dim=0)

                if spk_lbl is None:
                    spk_lbl = i.get('spk_ids').repeat(lpv.size(1))
                else:
                    spk_lbl = torch.cat([spk_lbl, i.get('spk_ids').repeat(lpv.size(1))], dim=0)                
                # break
            print("Num LPV: ", lpvs.shape, word_lbl.shape, spk_lbl.shape)

            lpvs = lpvs.detach().cpu().numpy()
            word_lbl = word_lbl.detach().cpu().numpy()
            spk_lbl = spk_lbl.detach().cpu().numpy()
            np.save("{}/lpvs.npy".format(hparams["work_dir"]), lpvs)
        
        print("########## Extracting LPV: ", lpvs.shape)
        kmeans = MiniBatchKMeans(n_clusters=128, 
                    init='k-means++', 
                    max_iter=300,
                    batch_size=10000,
                    random_state=0,
                    tol=0.0,
                    reassignment_ratio=0.5,
                    )

        kmeans.fit(lpvs)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = torch.from_numpy(cluster_centers)
        print("K-Means Done")
        self.model.prosody_encoder.init_vq(cluster_centers)
        self.model.train()
        self.is_lpv_init = True

    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']
        word_tokens = sample['word_tokens']
        spk_embed = sample.get('spk_embed')

        if not infer and self.global_step == hparams["vq_warmup"] and not self.is_lpv_init:
            self.cluster_and_init()
        
        if not infer:
            output = self.model(txt_tokens, word_tokens,
                                ph2word=sample['ph2word'],
                                mel2word=sample['mel2word'],
                                mel2ph=sample['mel2ph'],
                                word_len=sample['word_lengths'].max(),
                                tgt_mels=sample['mels'],
                                spk_embed=spk_embed,
                                infer=False,
                                global_step=self.global_step
                                )
            losses = {}

            if self.global_step > hparams["vq_warmup"]:
                losses["vq_loss"] = output["vq_loss"]
            self.add_mel_loss(output['mel_out'], sample['mels'], losses)
            super(DiffProsodyTask, self).add_dur_loss(output['dur'], sample['mel2ph'], sample['txt_tokens'], losses)
            return losses, output
        else:
            output = self.model(
                txt_tokens, word_tokens,
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                mel2ph=sample['mel2ph'],
                mel2word=sample['mel2word'],
                tgt_mels=sample['mels'],
                infer=False,
                spk_embed=spk_embed,
                global_step=self.global_step,
            )
            return output

    def _training_step(self, sample, batch_idx, optimizer_idx):

        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and (hparams['lambda_mel_adv'] > 0. or hparams['lambda_cond_adv'] > 0.)

        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}

            if self.global_step > hparams["vq_warmup"]:
                loss_output['perplexity'] = model_out['perplexity']

            if disc_start:
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                lpv = model_out['lpv_long']
                o = self.mel_disc(mel_g, lpv)
                o_ = self.mel_disc(mel_p, lpv)

                p = o['y']
                p_ = o_['y']

                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']

            total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
            loss_output['total_loss'] = total_loss
            loss_output['batch_size'] = sample['txt_tokens'].size()[0]
            
            return total_loss, loss_output
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out'].detach()
                lpv = model_out['lpv_long'].detach()
                o = self.mel_disc(mel_g, lpv)
                o_ = self.mel_disc(mel_p, lpv)

                p = o['y']
                p_ = o_['y']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))

                total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
                loss_output['total_loss_d'] = total_loss
                loss_output['batch_size'] = sample['txt_tokens'].size()[0]
                
                return total_loss, loss_output
            total_loss = torch.Tensor([0]).float()
            return total_loss, loss_output

    def plot_histogram(self, x):
        fig = plt.figure(figsize=(12, 6))
        plt.hist(x, bins=range(128))
        plt.grid(True)
        return fig

    def add_dur_loss(self, dur_pred, mel2token, word_len, txt_tokens, losses=None):
        T = word_len.max()
        dur_gt = mel2token_to_dur(mel2token, T).float()
        nonpadding = (torch.arange(T).to(dur_pred.device)[None, :] < word_len[:, None]).float()
        dur_pred = dur_pred * nonpadding
        dur_gt = dur_gt * nonpadding
        wdur = F.l1_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        wdur = (wdur * nonpadding).sum() / nonpadding.sum()
        if hparams['lambda_word_dur'] > 0:
            losses['wdur'] = wdur * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.l1_loss(sent_dur_p, sent_dur_g, reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']

    def validation_step(self, sample, batch_idx):        
        return super(DiffProsodyTask, self).validation_step(sample, batch_idx)

    def save_valid_result(self, sample, batch_idx, model_out):
        super(DiffProsodyTask, self).save_valid_result(sample, batch_idx, model_out)

        if self.global_step > hparams["vq_warmup"]:
            lpv = model_out["lpv_idx"].cpu().detach().numpy()
            if batch_idx == 0:
                self.lpv_dist = lpv
            elif batch_idx < (hparams["num_valid_plots"] - 1):
                self.lpv_dist = np.concatenate((self.lpv_dist, lpv), axis=0)
            elif batch_idx == (hparams["num_valid_plots"] - 1):
                self.lpv_dist = np.concatenate((self.lpv_dist, lpv), axis=0)
                self.logger.add_figure(f'lpv_histogram', 
                        self.plot_histogram(self.lpv_dist),
                        self.global_step)

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])

    ############
    # infer
    ############
    def test_start(self):
        super().test_start()

    def test_step(self, sample, batch_idx):
        assert sample['txt_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        outputs = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        mel2ph = sample['mel2ph'][0].cpu().numpy()
        mel2ph_pred = None
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'
        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred])
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph])
        if hparams.get('save_attn', False):
            attn = outputs['attn'][0].cpu().numpy()
            np.save(f'{gen_dir}/attn/{item_name}.npy', attn)
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }
