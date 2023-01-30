### DiffProsody: Latent Diffusion-based Prosody Generation for Expressive Speech Synthesis with Prosody Conditional Adversarial Training [[Demo]](https://prml-lab-speech-team.github.io/demo/DiffProsody/)

#### H.-S. Oh, S.-H. Lee, S.-W. Lee, *ICASSP*, 2023 (submitted)

## Abstract

Expressive text-to-speech systems have undergone significant improvements with prosody modeling. However, there is still room for improvement in the conventional prosody modeling methods. Previous studies have used the autoregressive method to predict the quantized prosody vector, which has long-term dependency and slow inference problems. In this paper, we present DiffProsody, which can synthesize high-quality expressive speech using a latent diffusion-based prosody generator and prosody conditional adversarial training to solve these problems. We show that the prosody generator can generate high-quality prosody vector and the prosody conditional discriminator can improve the quality of generated speech by reflecting the prosody. Furthermore, we adopted denoising diffusion generative adversarial networks to improve the prosody generation speed. Hence, DiffProsody can generate prosody 16x faster than the conventional diffusion model. The experimental results demonstrated the superiority of the proposed method.

## Model
![image](assets/model.png)

## Training Procedure
### Environments
```
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
bash mfa_usr/install_mfa.sh # install force alignment tools
```

### 1. Preprocess data

- Download [VCTK](https://datashare.ed.ac.uk/handle/10283/2651) dataset

```bash
# Preprocess step: text and unify the file structure.
python data_gen/tts/runs/preprocess.py --config "egs/datasets/audio/vctk/diffprosody.yaml"
# Align step: MFA alignment.
python data_gen/tts/runs/train_mfa_align.py --config "egs/datasets/audio/vctk/diffprosody.yaml"
# Binarization step: Binarize data for fast IO. You only need to rerun this line when running different task if you have `preprocess`ed and `align`ed the dataset before.
python data_gen/tts/runs/binarize.py --config "egs/datasets/audio/vctk/diffprosody.yaml"
```

### 2. Training TTS module and prosody encoder  
```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config "egs/datasets/audio/vctk/diffprosody.yaml" --exp_name "DiffProsody"
```

### 3. Extracting latent prosody vector 
```bash
CUDA_VISIBLE_DEVICES=0 python extract_lpv.py --config "egs/datasets/audio/vctk/diffprosody.yaml" --exp_name "DiffProsody"
```

### 4. Training diffusion-based latent prosody generator
- You should set the path according to your environment
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config "egs/datasets/audio/vctk/prosody_generator.yaml" --exp_name "DiffProsodyGenerator" --reset --hparams="tts_model=/{ckpt dir}/DiffProsody"
```

### 4. Inference

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config "egs/datasets/audio/vctk/prosody_generator.yaml" --exp_name "DiffProsodyGenerator" --infer --hparams="tts_model=/{ckpt dir}/DiffProsody"
```

### 5. Pretrained checkpoints
- TTS module trained on 160k [[Download]](https://works.do/xsBlIw8)
- Diffusion-based prosody generator trained on 320k [[Download]](https://works.do/5CAF6E0)

## Acknowledgements
**Our codes are based on the following repos:**
* [NATSpeech](https://github.com/NATSpeech/NATSpeech)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [HifiGAN](https://github.com/jik876/hifi-gan)