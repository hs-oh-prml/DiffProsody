import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.tts.diffprosody.diffprosody import DiffProsody
from modules.tts.diffprosody.diffusion import DiffusionProsodyGenerator
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams
from tqdm import tqdm 
from utils.nn.seq_utils import group_hidden_by_segs
import os 
from utils.audio.io import save_wav
from resemblyzer import VoiceEncoder
from data_gen.tts.txt_processors.base_text_processor import get_txt_processor_cls
from utils.text.text_encoder import build_token_encoder
import librosa 
from speechbrain.pretrained import EncoderClassifier

class DiffProsodyInfer(BaseTTSInfer):
    def build_model(self):
        ph_dict_size = len(self.ph_encoder)
        word_dict_size = len(self.word_encoder)
        model = DiffProsody(ph_dict_size, word_dict_size, self.hparams)
        model.prosody_encoder.init_vq(None)
        load_ckpt(model, hparams['tts_model'], 'model')
        model.eval()
        self.pd = DiffusionProsodyGenerator(self.hparams)
        load_ckpt(self.pd, hparams['work_dir'], 'model')
        self.pd.cuda()
        self.pd.eval()

        txt_processor = self.preprocess_args['txt_processor']
        self.txt_processor = get_txt_processor_cls(txt_processor)
        self.ph_encoder = build_token_encoder('/workspace/dataset/binary/vctk/phone_set.json')
        self.word_encoder = build_token_encoder('/workspace/dataset/binary/vctk/word_set.json')
        self.resem = VoiceEncoder().cuda()
        return model

    def forward_model(self, inp):
        items = inp.rstrip().split("|")
        
        wav_name = items[0]
        txt_raw = items[1]

        txt_struct, txt = self.txt_processor.process(txt_raw, hparams['preprocess_args'])
        ph = [p for w in txt_struct for p in w[1]]
        words = [w[0] for w in txt_struct]
        ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
        ph = " ".join(ph)
        word = " ".join(words)

        word_token = self.word_encoder.encode(word)
        ph_token = self.ph_encoder.encode(ph)

        infer_dir = os.path.join(hparams['work_dir'], "infer")
        os.makedirs(infer_dir, exist_ok=True)

        wav_dir = "/workspace/dataset/libritts/wavs"

        with torch.no_grad():
            y, _ = librosa.load(os.path.join(wav_dir, wav_name+'.wav'))
            y, _ = librosa.effects.trim(y)
            resem = self.resem.embed_utterance(y.astype(float))
            y = torch.Tensor(y).to("cuda:0")
            y = y.to("cuda:0")
            spk_embed = torch.Tensor(resem).unsqueeze(0).cuda()
            spk_embed = self.model.forward_style_embed(spk_embed, None).squeeze(1)

            txt_tokens = torch.LongTensor(ph_token).unsqueeze(0).cuda()
            ph2word = torch.LongTensor(ph2word).unsqueeze(0).cuda()
            word_tokens = torch.LongTensor(word_token).unsqueeze(0).cuda()
            word_lengths = len(word_token)

            h_ling = self.model.run_text_encoder(txt_tokens, word_tokens, 
                                                ph2word, word_lengths, 
                                                None, None, {})
            h_ling = group_hidden_by_segs(h_ling, ph2word, word_lengths)[0]
        
            wrd_nonpadding = (word_tokens > 0).float()
            output = self.pd(h_ling, 
                                spk_embed=spk_embed,
                                ph2word=ph2word,
                                infer=True,
                                padding=wrd_nonpadding)
            lpv = self.model.prosody_encoder.vector_quantization(output["lpv_out"])[1]
            output = self.model(
                txt_tokens,
                word_tokens,
                ph2word=ph2word,
                word_len=word_lengths,
                infer=True,
                spk_embed=spk_embed.squeeze(1),
                lpv=lpv
            )

            mel = output['mel_out'][0].cpu().numpy()
            wav = self.vocoder.spec2wav(mel)
            n = wav_name
            save_wav(wav, os.path.join(infer_dir, n+".wav"), 
                            hparams['audio_sample_rate'],
                            norm=hparams['out_wav_norm'])


    def infer_once(self, inp):
        output = self.forward_model(inp)
        return output

    @classmethod
    def example_run(cls):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp

        set_hparams()
        
        f = open("./zeroshot_libri.txt", "r")
        lines = f.readlines()
        infer_ins = cls(hp)
        for idx, i in tqdm(enumerate(lines)):
            infer_ins.infer_once(i)
            # if idx > 10: break
if __name__ == '__main__':
    DiffProsodyInfer.example_run()
