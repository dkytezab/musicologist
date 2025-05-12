import yaml
import imageio.v2 as imageio
import io
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os

from concept_datasets import ConceptDataset, cache_model
from models import BinaryClassifier
from concept_filters import get_all_concepts

with open('interp/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

embed_model = config["embed_model"]
class_models = config["class_models"]
truncation_ts = config["truncation_ts"]
overwrite = config["overwrite"]

hparams = {k: v for c in config["hparams"] for k, v in c.items()}

results_dict = {}
limit_dict = {}

# Change for specific concepts if desired
concepts = get_all_concepts()


def process_embeds() -> None:
    '''First caches laion-clap; then partitions NSynth train to sample positive, negative concept samples; then gets the embeddings
    for the selected embeddings which are saved.

    To generate a new partition/set of embeddings, set overwrite = True. As a warning this is somewhat computationally hard as 
    NSynth train is fairly large.
    '''
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
    'Trains binary classifiers, iterating over all classifier model types (logistic, svm) and all concepts.'
    for concept in concepts:
        for class_model in class_models:
            hparams["model_type"] = class_model

            # Initializes binary classifier
            binary_classifier = BinaryClassifier(
                        concept_filter=concept,
                        num_pos_samples=limit_dict[concept][0],
                        num_neg_samples=limit_dict[concept][1],
                        model_name=embed_model,
                        hparams=hparams,
                    )
            
            # Trains binary classifier using default csv path.
            binary_classifier.train(csv_path=None)

            # Inference for each batch of diffusion steps
            for diff_step in truncation_ts:
                binary_classifier.inference(diff_step=diff_step, embed_model=embed_model, save=True)
                
                # Only logistic PCA gif 
                if class_model=="logistic":
                    ims = binary_classifier.get_pca(diff_steps=truncation_ts, embed_model=embed_model)

                    figs = [fig_to_pil(im) for im in ims]
                    figs[0].save(
                        f'data/concepts/{concept}/pca_{class_model}.gif',
                        save_all=True,
                        append_images=figs[1:],
                        duration=500,
                        loop=0 
                        )

def fig_to_pil(fig):
    'Helper for creating gif'
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def results_to_im():
    '''Produces graph of the accuracy, true positive rate and true negative rate for a given concept
    taken over the diffusion process. Saves 2 separate figures in the relevant concept's folder.
    '''
    for concept in concepts:
        json_path = f"data/concepts/{concept}/concept_results.json"
        df = pd.read_json(json_path)

        # Getting computed results
        log_acc = df.loc[df["class_model_type"] == "logistic", "acc"]
        log_tpr = df.loc[df["class_model_type"] == "logistic", "tpr"]
        log_tnr = df.loc[df["class_model_type"] == "logistic", "tnr"]

        svm_acc = df.loc[df["class_model_type"] == "svm", "acc"]
        svm_tpr = df.loc[df["class_model_type"] == "svm", "tpr"]
        svm_tnr = df.loc[df["class_model_type"] == "svm", "tnr"]

        # Number of positive, negative samples
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

        # Making plots look nice :D
        for k in [ax1, ax2]:
            k.set_xticks(truncation_ts)
            k.set_xlabel("Diffusion Timestep")
            k.set_ylim(0, 1)
            k.set_yticks(np.arange(0, 1, 0.1))
            k.set_title(f"{concept} classifier performance on diffusion-generated audio with {num_pos} positive samples, {num_neg} negative samples")
            k.legend(loc='best')
            k.grid()

        # Including base performance on NSynth
        fig1.text(0.5, 0, f"Concept train accuracy: {round(log_train_acc, 3)}, Concept test accuracy: {round(log_test_acc, 3)}",
             ha='center', va='bottom', fontsize=9)
        fig2.text(0.5, 0, f"Concept train accuracy: {round(svm_train_acc, 3)}, Concept test accuracy: {round(svm_test_acc, 3)}",
             ha='center', va='bottom', fontsize=9)
        
        fig1.tight_layout()
        fig2.tight_layout()
        
        fig1.savefig(f"data/concepts/{concept}/logistic_results.png", dpi=300, bbox_inches="tight")
        fig2.savefig(f"data/concepts/{concept}/svm_results.png", dpi=300, bbox_inches="tight")

def get_clap_tensor_distance():
    'Computes and plots the tensor L1, L2 and cosine similarity for CLAP embeddings'
    tensors = []
    distances = {}

    # Getting CLAP tensor distance
    for diff_step in truncation_ts:
        clap_tens = torch.load(f"data/generated/diff_step_{diff_step}/{embed_model}_embeddings.pt")
        tensors.append(clap_tens)
    
    l1_norm, l2_norm, cosine_similarity = [], [], []
    for i in range(9):
        a, b = tensors[i], tensors[i+1]

        l1 = torch.norm(a - b, p=1).item()
        l1_norm.append(l1)

        l2 = torch.norm(a - b, p=2).item()
        l2_norm.append(l2)

        cos_sim = F.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=1).item()
        cosine_similarity.append(cos_sim)

    distances["L1"] = l1_norm
    distances["L2"] = l2_norm
    distances["Cosine Similarity"] = cosine_similarity

    for metric, values in distances.items():
        plt.figure()
        plt.xlabel("Pair index")
        plt.plot(range(1, len(values) + 1), values)
        plt.title(f"{metric.replace('_', ' ').title()} Between Successive Tensors")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.tight_layout()

        out_path = os.path.join(f"data/concepts", f"CLAP_{metric}.png")
        plt.savefig(out_path)
        plt.close()

# Main function
if __name__ == "__main__":
    process_embeds()
    interp()
    results_to_im()
    get_clap_tensor_distance()