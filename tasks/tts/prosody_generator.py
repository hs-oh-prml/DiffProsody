import os
import torch
import torch.nn.functional as F
from torch import nn

from modules.tts.diffprosody.diffprosody import DiffProsody
from modules.tts.diffprosody.diffusion import DiffusionProsodyGenerator
from modules.tts.diffprosody.discriminator import DiffDiscriminator
from tasks.tts.dataset_utils import DiffProsodyDataset

from tasks.tts.fs import FastSpeechTask
from utils.commons.hparams import hparams
from utils.nn.model_utils import num_params
from utils.text.text_encoder import build_token_encoder
import os 
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.tensor_utils import tensors_to_scalars
from utils.nn.seq_utils import group_hidden_by_segs

class ProsodyGeneratorTask(FastSpeechTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = DiffProsodyDataset
        data_dir = hparams['binary_data_dir']
        self.word_encoder = build_token_encoder(f'{data_dir}/word_set.json')
        self.n_layers = hparams["n_layer"] + hparams["n_cond_layer"]
    
    def build_tts_model(self):
        self.model = DiffusionProsodyGenerator(hparams)
        self.discriminator = DiffDiscriminator(hparams)
        self.disc_params = list(self.discriminator.parameters())

        ph_dict_size = len(self.token_encoder)
        word_dict_size = len(self.word_encoder)
        with torch.no_grad():
            self.tts = DiffProsody(ph_dict_size, word_dict_size, hparams)
            if os.path.exists("{}/lpvs.npy".format(hparams["tts_model"])):
                self.tts.prosody_encoder.init_vq(None)
            load_ckpt(self.tts, hparams['tts_model'], 'model')

            self.tts.eval()

    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        num_params(self.discriminator, model_name='disc')
        
    def run_model(self, sample, infer=False, *args, **kwargs):

        txt_tokens = sample['txt_tokens']
        ph2word = sample['ph2word']
        spk_id = sample.get('spk_ids')
        spk_embed = sample.get('spk_embed')
        spk_embed = self.tts.forward_style_embed(spk_embed, spk_id).squeeze(1)
        word_lengths = sample['word_lengths'] 
        
        word_tokens = sample['word_tokens']            
        h_ling = self.tts.run_text_encoder(txt_tokens, word_tokens, 
                                            ph2word, word_lengths.max(), 
                                            None, None, {})
        h_ling = group_hidden_by_segs(h_ling, ph2word, word_lengths.max())[0]
        
        wrd_nonpadding = (word_tokens > 0).float()

        if not infer:
        
            lpv = sample['lpvs']

            output = self.model(h_ling, 
                                spk_embed=spk_embed, 
                                lpv=lpv,
                                ph2word=ph2word,
                                infer=infer,
                                padding=wrd_nonpadding)
        else:
            output = self.model(h_ling, 
                                spk_embed=spk_embed,
                                ph2word=ph2word,
                                infer=infer,
                                padding=wrd_nonpadding)
            vq_lpv = self.tts.prosody_encoder.vector_quantization(output["lpv_out"])
            output["vq_lpv_out"] = vq_lpv[1]
                
        return output

    def _training_step(self, sample, batch_idx, optimizer_idx):
        
        loss_output = {}
        loss_weights = {}

        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            model_out = self.run_model(sample)

            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}

            x_0_pred = model_out['x_0_predict']
            x_t = model_out['x_t']
            x_tp1 = model_out['x_tp1']
            x_pos_sample = model_out['x_pos_sample']
            t = model_out['t']
            wrd_nonpadding = (sample['word_tokens'] > 0).float()

            cond = model_out['cond']
            lpv_gt = self.model.norm_spec(sample['lpvs'])
            lpv_pred = x_0_pred

            if hparams['lambda_lpv'] > 0.:                
                if hparams['lpv_loss'] == "l1":
                    lpv_loss = F.l1_loss(lpv_pred, lpv_gt, reduction="none")
                if hparams['lpv_loss'] == "mse":
                    lpv_loss = F.mse_loss(lpv_pred, lpv_gt, reduction="none")

                lpv_loss = lpv_loss * wrd_nonpadding.unsqueeze(-1)
                loss_output["lpv_loss"] = lpv_loss.sum() / wrd_nonpadding.sum()
                loss_weights['lpv_loss'] = hparams['lambda_lpv']

            D_fake_cond = self.discriminator(x_pos_sample, t, x_tp1, cond, wrd_nonpadding)                

            if hparams['lambda_adv'] > 0.:
                adv_loss = self.g_loss_fn(D_fake_cond[-1])
                loss_output["adv_loss"] = adv_loss
                loss_weights['adv_loss'] = hparams['lambda_adv']
            if hparams['lambda_fm'] > 0.:
                D_real_cond = self.discriminator(x_t, t, x_tp1, cond, wrd_nonpadding)
                fm_loss = self.fm_loss(D_real_cond, D_fake_cond)
                loss_output["fm_loss"] = fm_loss
                loss_weights['fm_loss'] = hparams['lambda_fm']

            total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])

            loss_output['total_loss'] = total_loss
            loss_output['batch_size'] = sample['txt_tokens'].size()[0]
            
            return total_loss, loss_output
        else:
            #######################
            #    Discriminator    #
            #######################
            if hparams['lambda_adv'] > 0.:
                    
                model_out = self.model_out_gt
                
                wrd_nonpadding = (sample['word_tokens'] > 0).float()
                
                x_0_pred = model_out['x_0_predict'].detach()
                x_t = model_out['x_t'].detach()
                x_tp1 = model_out['x_tp1'].detach()
                x_pos_sample = model_out['x_pos_sample'].detach()
                t = model_out['t'].detach()
                cond = model_out['cond'].detach()

                D_fake_cond = self.discriminator(x_pos_sample, t, x_tp1, cond, wrd_nonpadding)                
                D_real_cond = self.discriminator(x_t, t, x_tp1, cond, wrd_nonpadding)
                D_loss_real, D_loss_fake = self.d_loss_fn(D_real_cond[-1], D_fake_cond[-1])

                loss_output['D_loss_real'] = D_loss_real
                loss_output['D_loss_fake'] = D_loss_fake

                total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
                
                loss_output['total_loss_d'] = total_loss
                loss_output['batch_size'] = sample['txt_tokens'].size()[0]
                return total_loss, loss_output

            total_loss = torch.Tensor([0]).float()
            return total_loss, loss_output

    def jcu_loss_fn(self, logit, label_fn, mask=None):
        loss = F.mse_loss(logit, label_fn(logit), reduction="none" if mask is not None else "mean")
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss
        return loss 

    def d_loss_fn(self, r_logit_cond, f_logit_cond, mask=None):
        r_loss = self.jcu_loss_fn(r_logit_cond, torch.ones_like, mask)
        f_loss = self.jcu_loss_fn(f_logit_cond, torch.zeros_like, mask)
        return r_loss, f_loss

    def g_loss_fn(self, f_logit_cond, mask=None):
        f_loss = self.jcu_loss_fn(f_logit_cond, torch.ones_like, mask)
        return f_loss

    def fm_loss(self, D_real_cond, D_real_uncond, D_fake_cond, D_fake_uncond):
        loss_fm = 0
        feat_weights = 4.0 / (self.n_layers + 1)
        for j in range(len(D_fake_cond) - 1):
            loss_fm += feat_weights * \
                0.5 * (F.l1_loss(D_real_cond[j].detach(), D_fake_cond[j]) + F.l1_loss(D_real_uncond[j].detach(), D_fake_uncond[j]))
        return loss_fm

    def validation_step(self, sample, batch_idx):

        outputs = {}
        outputs['losses'] = {}
        outputs['nsamples'] = sample['nsamples']
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            pred_output = self.run_model(sample, infer=True)
            lpv_gt = sample['lpvs']
            lpv_gt = self.tts.prosody_encoder.vector_quantization(lpv_gt)[1]

            gt_mel = self.tts(
                    sample['txt_tokens'], sample['word_tokens'],
                    ph2word=sample['ph2word'],
                    word_len=sample['word_lengths'].max(),
                    infer=True,
                    spk_embed=sample['spk_embed'].squeeze(1),
                    lpv=lpv_gt,
                )

            lpv_pred = pred_output["vq_lpv_out"]
            pred_mel = self.tts(
                    sample['txt_tokens'], sample['word_tokens'],
                    ph2word=sample['ph2word'],
                    word_len=sample['word_lengths'].max(),
                    infer=True,
                    spk_embed=sample['spk_embed'].squeeze(1),
                    lpv=lpv_pred,
                )
            self.save_valid_result(sample, batch_idx, [gt_mel, pred_mel])
            
        outputs = tensors_to_scalars(outputs)
        return outputs

    def save_valid_result(self, sample, batch_idx, model_out):
        sr = hparams['audio_sample_rate']
        gt = model_out[0]
        pred = model_out[1]

        wav_title_gt = "gt_wav_{}".format(batch_idx)        
        wav_gt = self.vocoder.spec2wav(gt['mel_out'][0].cpu())
        self.logger.add_audio(wav_title_gt, wav_gt, self.global_step, sr)

        wav_title_pred = "pred_wav_{}".format(batch_idx)        
        wav_pred = self.vocoder.spec2wav(pred['mel_out'][0].cpu())
        self.logger.add_audio(wav_title_pred, wav_pred, self.global_step, sr)

        mel_title = "mel_{}".format(batch_idx)
        self.plot_mel(batch_idx, gt['mel_out'][0], pred['mel_out'][0], title=mel_title)

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
            FastSpeechTask.build_scheduler(self, optimizer[0]),     # Generator Scheduler
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
        assert sample['word_tokens'].shape[0] == 1, 'only support batch_size=1 in inference'
        pred_output = self.run_model(sample, infer=True)
        text = sample['text'][0]
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel2ph = None
        if hparams['use_gt_dur']:
            mel2ph = sample['mel2ph']
            
        lpv_pred = pred_output["vq_lpv_out"]
        
        tts_output = self.tts(
                    sample['txt_tokens'], sample['word_tokens'],
                    ph2word=sample['ph2word'],
                    word_len=sample['word_lengths'].max(),
                    infer=True,
                    spk_embed=sample['spk_embed'].squeeze(1),
                    lpv=lpv_pred,
                    mel2ph=mel2ph,
                )

        mel_pred = tts_output["mel_out"][0].cpu()
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        base_fn = f'[{batch_idx:06d}][{item_name.replace("%", "_")}][%s]'

        if text is not None:
            base_fn += text.replace(":", "$3A")[:80]
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred)

        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, None])
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, None])

        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.token_encoder.decode(tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }
