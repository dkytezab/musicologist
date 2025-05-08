import torch
import yaml

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
concepts = ['is_bass',]

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
                fig = binary_classifier.get_pca(diff_step=diff_step, embed_model=embed_model)
                fig.savefig(f"data/concepts/{concept}/PCA_{diff_step}", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    process_embeds()
    interp()
