import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.tts.diffprosody.diffprosody import DiffProsody
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams, set_hparams
from tqdm import tqdm 
from tasks.tts.tts_utils import load_data_preprocessor
from tasks.tts.dataset_utils import FastSpeechWordDataset
from utils.commons.tensor_utils import move_to_cuda
import os 
import numpy as np

class get_LPV(BaseTTSInfer):
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor, self.preprocess_args = load_data_preprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict(self.data_dir)
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        self.ds_cls = FastSpeechWordDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        
    def build_model(self):
        ph_dict_size = len(self.ph_encoder)
        word_dict_size = len(self.word_encoder)
        model = DiffProsody(ph_dict_size, word_dict_size, self.hparams)
        if os.path.exists("{}/lpvs.npy".format(hparams["work_dir"])):
            model.prosody_encoder.init_vq(None)
        load_ckpt(model, hparams['work_dir'], 'model')
        model.eval()
        return model

    def forward_model(self, i):
        with torch.no_grad():
            i = move_to_cuda(i, self.device)
            lpv, lpv_idx = self.model.get_lpv(
                    i["txt_tokens"], i["word_tokens"],
                    i["ph2word"], i["word_lengths"].max(),
                    i["mel2word"], i["mel2ph"],
                    i.get('spk_embed'),
                    i['mels'], 160000
                )
        return lpv, lpv_idx

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls):
        from utils.commons.hparams import hparams as hp
        import json 
        dataset_cls = FastSpeechWordDataset
        l = [hparams['train_set_name'], hparams['valid_set_name'], hparams['test_set_name']]
        # l = [hparams['test_set_name']]
        
        lpv_dir = os.path.join(hparams['work_dir'], "lpvs")

        os.makedirs(lpv_dir, exist_ok=True)
        for d in l:
            dataset = dataset_cls(prefix=d, shuffle=True)
            dataloader = torch.utils.data.DataLoader(dataset,
                                        collate_fn=dataset.collater,
                                        batch_size=1,     # Numer of Sample 
                                        num_workers=1,
                                        pin_memory=False)
            print("###### Dataloader: ", len(dataloader))
            infer_ins = cls(hp)
            
            lpv_min = None
            lpv_max = None

            for idx, i in tqdm(enumerate(dataloader)):
                lpv, out_idx = infer_ins.infer_once(i)

                j_obj = {
                    "lpv": lpv.cpu().numpy()[0],
                    "lpv_idx": out_idx.cpu().numpy()[0]
                }
                
                npz_name = os.path.join(lpv_dir, i["item_name"][0]+".npz")
                np.savez(npz_name, **j_obj)

                lpv = lpv.cpu().numpy().tolist()
                lpv = np.array(lpv).squeeze(0)
                if lpv_min is None:
                    lpv_min = lpv
                    lpv_max = lpv 
                temp = np.concatenate((lpv_min, lpv), axis=0)
                temp2 = np.concatenate((lpv_max, lpv), axis=0)
                lpv_min = np.min(temp, axis=0)
                lpv_max = np.max(temp2, axis=0)
                lpv_min = np.expand_dims(lpv_min, axis=0)
                lpv_max = np.expand_dims(lpv_max, axis=0)
            j_obj = {
                "lpv_min": lpv_min.tolist(),
                "lpv_max": lpv_max.tolist(),
            }
            with open(os.path.join(hparams['work_dir'], "stats_lpv_{}.json".format(d)), "w") as w:
                json.dump(j_obj, w)

if __name__ == '__main__':
    set_hparams()
    get_LPV.example_run()
