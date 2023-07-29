### DiffProsody: Latent Diffusion-based Prosody Generation for Expressive Speech Synthesis with Prosody Conditional Adversarial Training [[Demo]](https://prml-lab-speech-team.github.io/demo/DiffProsody/)

## Abstract

Expressive text-to-speech systems have undergone significant advancements owing to prosody modeling, but conventional methods can still be improved. Traditional approaches have relied on the autoregressive method to predict the quantized prosody vector; however, it suffers from the issues of long-term dependency and slow inference. This study proposes a novel approach called DiffProsody in which expressive speech is synthesized using a diffusion-based latent prosody generator and prosody conditional adversarial training. Our findings confirm the effectiveness of our prosody generator in generating a prosody vector. Furthermore, our prosody conditional discriminator significantly improves the quality of the generated speech by accurately emulating prosody. We use denoising diffusion generative adversarial networks to improve the prosody generation speed. Consequently, DiffProsody is capable of generating prosody 16 times faster than the conventional diffusion model. The superior performance of our proposed method has been demonstrated via experiments.

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

### 5. Inference

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config "egs/datasets/audio/vctk/prosody_generator.yaml" --exp_name "DiffProsodyGenerator" --infer --hparams="tts_model=/{ckpt dir}/DiffProsody"
```

### 6. Pretrained checkpoints
- TTS module trained on 160k [[Download]](https://works.do/xsBlIw8)
- Diffusion-based prosody generator trained on 320k [[Download]](https://works.do/5CAF6E0)

## Acknowledgements
**Our codes are based on the following repos:**
* [NATSpeech](https://github.com/NATSpeech/NATSpeech)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [HifiGAN](https://github.com/jik876/hifi-gan)
