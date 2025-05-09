import torch
import yaml
import imageio.v2 as imageio
import io
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from concept_datasets import ConceptDataset, cache_model
from models import BinaryClassifier
from concept_filters import get_all_concepts

with open('interp/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

embed_model = config["embed_model"]
class_models = config["class_models"]
concepts = config["concepts"]
truncation_ts = config["truncation_ts"]
overwrite = config["overwrite"]

hparams = {k: v for c in config["hparams"] for k, v in c.items()}

results_dict = {}
limit_dict = {}

concepts = get_all_concepts()
# concepts = ['is_electronic', 'is_synthetic', 'is_orchestral', 'is_acoustic_band', 'is_percussion', 'is_techno', 'is_plucked', 'is_blown', 'is_hit', 'is_atmospheric']


def process_embeds():

    print(f"Started caching {embed_model}...")
    cache = cache_model(model_name=embed_model)
    print(f"...Finished caching {embed_model}")

    for concept in concepts:

        print(f"Processing {concept} dataset")
        cds = ConceptDataset(split="train", concept_filter=concept,
                             pos_limit=None, neg_limit=None, overwrite=overwrite)
        
        limit_dict[concept] = (cds.pos_limit, cds.neg_limit)
        
        print(f"Getting {concept} embeddings from {embed_model}")
        cds.get_embeds(model_name=embed_model, save=True, cache=cache)

def interp():
    for concept in concepts:
        for class_model in class_models:
            hparams["model_type"] = class_model

            binary_classifier = BinaryClassifier(
                        concept_filter=concept,
                        num_pos_samples=limit_dict[concept][0],
                        num_neg_samples=limit_dict[concept][1],
                        model_name=embed_model,
                        hparams=hparams,
                    )
            
            binary_classifier.train(csv_path=None)

            for diff_step in truncation_ts:
                binary_classifier.inference(diff_step=diff_step, embed_model=embed_model, save=True)
                
                # Only logistic PCA gif for computational ease
                if class_model=="logistic":
                    ims = binary_classifier.get_pca(diff_steps=truncation_ts, embed_model=embed_model)

                    figs = [fig_to_pil(im) for im in ims]
                    figs[0].save(
                        f'data/concepts/{concept}/pca_{class_model}.gif',
                        save_all=True,
                        append_images=figs[1:],
                        duration=500,  # ms per frame
                        loop=0  # loop forever
                        )

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

def results_to_im():
    for concept in concepts:
        json_path = f"data/concepts/{concept}/concept_results.json"
        df = pd.read_json(json_path)

        log_acc = df.loc[df["class_model_type"] == "logistic", "acc"]
        log_tpr = df.loc[df["class_model_type"] == "logistic", "tpr"]
        log_tnr = df.loc[df["class_model_type"] == "logistic", "tnr"]

        svm_acc = df.loc[df["class_model_type"] == "svm", "acc"]
        svm_tpr = df.loc[df["class_model_type"] == "svm", "tpr"]
        svm_tnr = df.loc[df["class_model_type"] == "svm", "tnr"]

        num_pos = df.iloc[0]["pos_gen_audio_samples"]
        num_neg = df.iloc[0]["neg_gen_audio_samples"]

        log_train_acc = df.loc[df["class_model_type"] == "logistic", "class_model_train_acc"].iloc[0]
        log_test_acc = df.loc[df["class_model_type"] == "logistic", "class_model_test_acc"].iloc[0]

        svm_train_acc = df.loc[df["class_model_type"] == "svm", "class_model_train_acc"].iloc[0]
        svm_test_acc = df.loc[df["class_model_type"] == "svm", "class_model_test_acc"].iloc[0]

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        ax1.plot(truncation_ts, log_acc, marker='o', color='black', label="Logistic Accuracy")
        ax1.plot(truncation_ts, log_tpr, marker='s', linestyle='--', color='green', label="Logistic TPR")
        ax1.plot(truncation_ts, log_tnr, marker='^', linestyle='--', color='firebrick', label='Logistic TNR')

        ax2.plot(truncation_ts, svm_acc, marker='o', color='black', label="SVM Accuracy")
        ax2.plot(truncation_ts, svm_tpr, marker='s', linestyle='--', color='green', label="SVM TPR")
        ax2.plot(truncation_ts, svm_tnr, marker='^', linestyle='--', color='firebrick', label='SVM TNR')


        for k in [ax1, ax2]:
            k.set_xticks(truncation_ts)
            k.set_xlabel("Diffusion Timestep")
            k.set_ylim(0, 1)
            k.set_yticks(np.arange(0, 1, 0.1))
            k.set_title(f"{concept} classifier performance on diffusion-generated audio with {num_pos} positive samples, {num_neg} negative samples")
            k.legend(loc='best')
            k.grid()

        fig1.text(0.5, 0, f"Concept train accuracy: {round(log_train_acc, 3)}, Concept test accuracy: {round(log_test_acc, 3)}",
             ha='center', va='bottom', fontsize=9)
        fig2.text(0.5, 0, f"Concept train accuracy: {round(svm_train_acc, 3)}, Concept test accuracy: {round(svm_test_acc, 3)}",
             ha='center', va='bottom', fontsize=9)
        
        fig1.tight_layout()
        fig2.tight_layout()
        

        fig1.savefig(f"data/concepts/{concept}/logistic_results.png", dpi=300, bbox_inches="tight")
        fig2.savefig(f"data/concepts/{concept}/svm_results.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # process_embeds()
    # interp()
    results_to_im()
