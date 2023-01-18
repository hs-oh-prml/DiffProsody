
### DiffProsody: Latent Diffusion-based Prosody Generation for Expressive Speech Synthesis with Prosody Conditional Adversarial Training [[Demo](https://prml-lab-speech-team.github.io/demo/DiffProsody/)]

#### H.-S. Oh, S.-H. Lee, S.-W. Lee, *ICASSP*, 2023 (submitted)

## Abstract

Expressive text-to-speech systems have undergone significant improvements with prosody modeling. However, there is still room for improvement in the conventional prosody modeling methods. Previous studies have used the autoregressive method to predict the quantized prosody vector, which has long-term dependency and slow inference problems. In this paper, we present DiffProsody, which can synthesize high-quality expressive speech using a latent diffusion-based prosody generator and prosody conditional adversarial training to solve these problems. We show that the prosody generator can generate high-quality prosody vector and the prosody conditional discriminator can improve the quality of generated speech by reflecting the prosody. Furthermore, we adopted denoising diffusion generative adversarial networks to improve the prosody generation speed. Hence, DiffProsody can generate prosody 16x faster than the conventional diffusion model. The experimental results demonstrated the superiority of the proposed method.

## Model
![image](assets/model.png)

## Run

```bash
bash run.sh
```

## Acknowledgements

**Our codes are based on the following repos:**

* [NATSpeech](https://github.com/NATSpeech/NATSpeech)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [HifiGAN](https://github.com/jik876/hifi-gan)
* [espnet](https://github.com/espnet/espnet)
