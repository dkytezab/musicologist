# Musicologist
Charting the emergence of concepts in diffusion-generated audio over a full trajectory. Our full pipeline consists of the following 6 steps:

(1) Generating music prompts and annotating them with a pre-defined set of concepts using GPT-4.1-mini.
(2) Generating batches of audio using Stable Diffusion 1.0 and saving intermediate steps drawn from the denoising process.
(3) Passing audio into CLAP to get representations.
(4) Creating datasets of positive-negative pairs from NSynth audio samples corresponding to pre-selected timbral concepts.
(5) Getting NSynth audio embeddings from CLAP
(6) Training a binary classifier on NSynth embeddings and performing inference on diffusion-generated audio embeddings.

Our code is structured as: 

``` text
├── data/
|   ├── concepts/
|   ├── generated/
|   ├── prompts/
├── diffusion/
├── embeddings/
├── interp/

Generated audio is stored at `data/generated/`, concepts embeddings and results are stored at `data/concepts` and code for (1) is stored at `data/prompts/`. `diffusion/` and `embeddings/` contain code for running (2) and (3) respectively. Code for (4), (5), (6) is stored at `interp/`. 

As our full pipeline is fairly complex, we provide instructions for re-creating our results starting at any one of steps (1)-(6) and relevant data, i.e. CLAP embeddings and CSV files. First run these lines from the root of the repo to create the virtual environment:
```bash
conda env create -f requirements.yaml
conda activate musicologist
```
To access Stable Audio 1.0, go to the [model card](https://huggingface.co/stabilityai/stable-audio-open-1.0) on HuggingFace and fill out the access form. Then run
```
huggingface-cli login
```
And enter your HuggingFace access token. Note that creating the environment may take a long time due to resolving dependencies for `stable_audio_tools`. If you wish to skip steps (1)-(2), please comment the following lines in `requirements.yaml`:
```bash
- fsspec==2024.3.1
- s3fs==2024.3.1
- stable_audio_tools
```


## Citations

| Citation | Description |
|----------|-------------|
| [[EVA2024]](https://arxiv.org/pdf/2402.04825) | Fast Timing-Conditioned Latent Audio Diffusion |
| [[GAN2025]](https://arxiv.org/pdf/2502.01639) | SliderSpace: Decomposing the Visual Capabilities of Diffusion Models |
| [[KIM2017]](https://arxiv.org/pdf/1711.11279) | Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) |
| [[WU2025]](https://arxiv.org/pdf/2211.06687) | Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation |
