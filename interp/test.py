from concept_datasets import ConceptDataset
from models import BinaryClassifier
import torch
import pandas as pd
import itertools

embeds = torch.load("data/concepts/is_brass/laion-clap_embeds.pt")
print(f"Embeddings shape: {embeds.shape}")

class1 = BinaryClassifier(
    concept_filter="is_brass",
    num_pos_samples=250,
    num_neg_samples=250,
    model_name="laion-clap",
    hparams={
        "model_type": "logistic",
        "test_size": 0.2
    }
)

class1.train(csv_path=None)

print(f"Train accuracy: {class1.concept_train_accuracy}")
print(f"Test accuracy: {class1.concept_test_accuracy}")
print(f"Model coefficients: {class1.model.coef_}")

diff_step = 49
pt_path = f"data/generated/diff_step_{diff_step}/laion-clap_embeddings.pt"

acc = class1.inference(pt_path=pt_path)

print(f"Audio accuracy for diff step {diff_step}: {acc}")

# d = pd.read_csv("data/generated/gen_audio_info.csv", low_memory=False)
# d.columns = d.columns.str.strip()

# a_vals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
# b_vals = range(1000) 
# c_vals = range(7)


# observed = set(zip(d['diffusion_step'], d['prompt_index'], d['sample_index']))
# for a,b,c in itertools.product(a_vals, b_vals, c_vals):
#     if (str(a),str(b),str(c)) not in observed:
#         print(a, b, c)


# brass_dataset = ConceptDataset(split="valid", concept_filter="is_blown", pos_limit=2400, neg_limit=8000, overwrite=True)
# embeds = brass_dataset.get_embeds(model_name="laion-clap", save=True)

