import torch
import yaml
import imageio.v2 as imageio
import io
from PIL import Image
import pandas as pd

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

# concepts = get_all_concepts()
concepts = ['is_electronic', 'is_synthetic', 'is_orchestral', 'is_acoustic_band', 'is_percussion', 'is_techno', 'is_plucked', 'is_blown', 'is_hit', 'is_atmospheric']


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
        json_path = f"data/generated/{concept}/concept_results.json"
        json = pd.read_json(json_path)


if __name__ == "__main__":
    process_embeds()
    interp()
