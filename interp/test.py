from concept_datasets import ConceptDataset
from models import BinaryClassifier
import torch
import pandas as pd
import itertools

pl = 10000
nl = 10000

string_dataset = ConceptDataset(split="train", concept_filter="is_string", pos_limit=pl, neg_limit=nl, overwrite=True)
embeds = string_dataset.get_embeds(model_name="laion-clap", save=True)

embeds = torch.load("data/concepts/is_string/laion-clap_embeds.pt")
print(f"Embeddings shape: {embeds.shape}")

class1 = BinaryClassifier(
    concept_filter="is_string",
    num_pos_samples=pl,
    num_neg_samples=nl,
    model_name="laion-clap",
    hparams={
        "model_type": "logistic",
        "test_size": 0.15
    }
)

class1.train(csv_path=None)

print(f"Train accuracy: {class1.concept_train_accuracy}")
print(f"Test accuracy: {class1.concept_test_accuracy}")
print(f"Model coefficients: {class1.model.coef_}")

diff_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
dict = {}
for step in diff_steps:
    path = f"data/generated/diff_step_{step}/laion-clap_embeddings.pt"
    tpr, tnr, acc = class1.inference(pt_path=path)
    dict[step] = (tpr, tnr, acc)
for step in diff_steps:
    print(f"TPR for diff step {step}: {dict[step][0]}")
    print(f"TNR for diff step {step}: {dict[step][1]}")
    print(f"Audio accuracy for diff step {step}: {dict[step][2]}")

# d = pd.read_csv("data/generated/gen_audio_info.csv", low_memory=False)
# d.columns = d.columns.str.strip()

# a_vals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
# b_vals = range(1000) 
# c_vals = range(7)


# observed = set(zip(d['diffusion_step'], d['prompt_index'], d['sample_index']))
# for a,b,c in itertools.product(a_vals, b_vals, c_vals):
#     if (str(a),str(b),str(c)) not in observed:
#         print(a, b, c)



