# Musicologist
Charting the emergence of concepts in diffusion-generated audio over a full trajectory. Our full pipeline consists of the following 6 steps:

(1) Generating music prompts and annotating them with a pre-defined set of concepts using GPT-4.1-mini.

(2) Generating batches of audio using Stable Diffusion 1.0 and saving intermediate steps drawn from the denoising process.

(3) Passing audio into CLAP to get representations.

(4) Creating datasets of positive-negative pairs from NSynth audio samples corresponding to pre-selected timbral concepts.

(5) Getting NSynth audio embeddings from CLAP.

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
```

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
And enter your HuggingFace access token. Note that creating the environment may take a long time due to resolving dependencies for `stable_audio_tools`. If you wish to skip steps (1)-(2), please comment out the following lines in `requirements.yaml`:
```bash
- fsspec==2024.3.1
- s3fs==2024.3.1
- stable_audio_tools
```
## (1) — Prompts

To generate new prompts, please run
```bash
python data/prompts/llm_promptgen.py
```
To annotate said prompts, attach one's `OPENAI_API_KEY` and then run
```bash
python data/prompts/annotate.py
```
Please note that this will use up one's OpenAI credits — we set our prompt batch size to 5 to avoid hallucinations + forgetting, which used up about one dollar in credits.

## (2) — Audio Generation

To generate the audio from prompts stored at `data/prompts/prompt.txt`, please run the following from the root of the repo:
```bash
module load cuda cudnn
make initgen
srun python diffusion/gen_distrib.py \
     --job-index $SLURM_ARRAY_TASK_ID \
     --num-jobs  $SLURM_ARRAY_TASK_COUNT
```
Make sure to include the constraint that all GPUs are Ampere GPUs, i.e. include `#SBATCH: --constraint=ampere`. Also include `#SBATCH --array=0-n` where `n - 1` is the total number of GPUs you want to divide the generation into. We found that delegating 4 prompts per GPU worked well and avoided OOM errors, so we set `n=249`. The results will save to `data/generated/diff_step_x` where `x` is the corresponding intermediate denoising step. 

Also be sure to clear `data/generated/audio_info.csv` of its past entries if you elect to generate new audio. This keeps track of all the important audio info.

## (3) - CLAP Embeddings

To generate CLAP embeddings from generated audio, run the following:

```bash
module load cuda cudnn
python embeddings/gen_embeds.py
```
Please keep the Ampere GPU constraint from part (2). The embeddings from audio in `data/generated/diff_step_x` will save to `data/generated/diff_step_x/laion-clap_embeddings`. 

## (4), (5), (6) - NSynth, Concepts, Classifiers

First create the directory `data/nsynth`. Then run
```bash
cd data/nsynth
curl --output nsynth-train.jsonwav.tar.gz "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
tar -xzvf nsynth-train.jsonwav.tar.gz
```
NSynth train is approximately 22 gigabytes, so installing all the data shouldn't take more than 40 minutes on the CPU. Then run
```bash
python interp/main.py
```
For each of the concepts in `interp/concept_filters.py`, we will create a concept dataset, store a CSV at `data/concepts/x`, get embeddings for said concept stored at `data/concepts/x/laion-clap_embeddings.py` and then train a logistic and svm classifier on the embeddings. A plot of the performance at inference time will be saved to `data/concepts/x`, as well as a PCA gif of the positive samples in the generated audio dataset.

If you instead want to run experiments for a select number of concepts, just change `concepts = get_all_concepts()` to a list of concepts of your choosing, i.e. `[x, ...]` where `x` is a key in `NSYNTH_FILTER_DICT`.

<!-- ## Citations

| Citation | Description |
|----------|-------------|
| [[EVA2024]](https://arxiv.org/pdf/2402.04825) | Fast Timing-Conditioned Latent Audio Diffusion |
| [[GAN2025]](https://arxiv.org/pdf/2502.01639) | SliderSpace: Decomposing the Visual Capabilities of Diffusion Models |
| [[KIM2017]](https://arxiv.org/pdf/1711.11279) | Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) |
| [[WU2025]](https://arxiv.org/pdf/2211.06687) | Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation | -->
