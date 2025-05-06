import torch
import yaml

from concept_datasets import ConceptDataset, get_pos_neg_limits
from models import BinaryClassifier

with open('interp/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

embed_model = config["embed_model"]
class_models = config["class_models"]
concepts = config["concepts"]
truncation_ts = config["truncation_ts"]
overwrite = config["overwrite"]

hparams = {k: v for c in config["hparams"] for k, v in c.items()}

results_dict = {}

def main():
    for concept in concepts:
        cds = ConceptDataset(split="train", concept_filter=concept,
                             pos_limit=None, neg_limit=None, overwrite=overwrite)
        
        pos_lim, neg_lim = cds.pos_limit, cds.neg_limit

        for class_model in class_models:
            hparams["model_type"] = class_model
            binary_classifier = BinaryClassifier(
                        concept_filter=concept,
                        num_pos_samples=pos_lim,
                        num_neg_samples=neg_lim,
                        model_name=embed_model,
                        hparams=hparams,
                    )
            binary_classifier.train(csv_path=None)

            concept_results = {
                "concept_train_accuracy": binary_classifier.concept_train_accuracy,
                "concept_test_accuracy": binary_classifier.concept_test_accuracy,
                "class_model": binary_classifier.model,
            }

            for step in truncation_ts:
                path = f"data/generated/diff_step_{step}/{embed_model}_embeddings.pt"
                tpr, tnr, acc = binary_classifier.inference(pt_path=path)
                
                gen_audio_dict = {
                    "tpr": tpr,
                    "tnr": tnr,
                    "accuracy": acc
                }

                concept_results[f"diff_step_{step}_results"] = gen_audio_dict

            results_dict[(concept, class_model)] = concept_results

    
    return results_dict

if __name__ == "__main__":
    results_dict = main()
    print(results_dict)