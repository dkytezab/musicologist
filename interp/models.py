from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.svm
from sklearn.metrics import recall_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

from concept_filters import GEN_AUDIO_FILTER_DICT, GEN_AUDIO_PADDER_DICT, create_concept_filter

class SimpleMLP(nn.Module):
    def __init__(self, hparams):
        super(SimpleMLP, self).__init__()

        self.num_layers = hparams.get("num_layers", 4)
        self.hidden_dim = hparams.get("hidden_dim", 512)
        self.input_dim = hparams.get("input_dim", 512)
        self.output_dim = hparams.get("output_dim", 1) 

        in_dim = self.input_dim
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim

        layers.append(nn.Linear(in_dim, self.output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def fit(self, 
            train_loader: DataLoader,
            epochs: int, 
            lr: float, 
            save_loss: bool = True,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            ):
        
        self.to(device).train()

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        epoch_losses = []
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    # ensure float and shape [N,1]
                    batch_y = batch_y.to(device).float().view(-1, 1)

                    logits = self(batch_x)
                    loss = criterion(logits, batch_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * batch_x.size(0)
            avg_loss = running_loss / len(train_loader.dataset)
            epoch_losses.append(avg_loss)

        return epoch_losses if save_loss else None
        
class BinaryClassifier(object):
    def __init__(self, 
                 concept_filter: str, 
                 num_pos_samples: int,
                 num_neg_samples: int,
                 model_name: str,
                 result_json_path: Optional[str] = None,
                 concept_pt_path: Optional[str] = None,
                 dir_path: Optional[str] = None,
                 hparams: Optional[Dict] = None,
                 ):

        self.concept_filter = concept_filter
        self.concept_padder = f"{concept_filter}_like"
        self.model_name = model_name

        self.dir_path = dir_path if dir_path is not None else f"data/concepts/{concept_filter}"  
        self.concept_pt_path = concept_pt_path if concept_pt_path is not None else f"{self.dir_path}/{model_name}_embeddings.pt"
        self.hparams = hparams if hparams is not None else {}

        self.num_pos_samples = num_pos_samples
        self.num_neg_samples = num_neg_samples

        self.result_json_path = result_json_path
        
    def train(self, csv_path: Optional[str] = None) -> None:
        csv_path = csv_path if csv_path is not None else f"{self.dir_path}/audio_info.csv"
        
        embeds, labels = self._load_concept_embeds(self.concept_pt_path, csv_path, self.model_name)

        if self.hparams["model_type"] == "logistic":
            tr_acc, te_acc, model = self._train_logistic_model(embeds, labels)
        elif self.hparams["model_type"] == "svm":
            tr_acc, te_acc, model = self._train_svm(embeds, labels)
        elif self.hparams["model_type"] == "mlp":
            tr_acc, te_acc, model = self._train_mlp(embeds, labels)
        else:
            raise NotImplementedError(f"Model type {self.hparams['model_type']} not implemented")

        self.concept_train_accuracy = tr_acc
        self.concept_test_accuracy = te_acc
        self.model = model
    
    def inference(self, diff_step: int, embed_model: str, save: bool = True,) -> None:
        pos_tensor, neg_tensor = self._load_gen_audio_embeds(csv_path=None, pt_path=None,
                                                             diff_step=diff_step, embed_model=embed_model)
        labels = np.concatenate([np.ones(pos_tensor.shape[0]), np.zeros(neg_tensor.shape[0])])
        input_tensor = torch.concat([pos_tensor, neg_tensor])

        if not self.hparams["model_type"] == "mlp":
            pred = self.model.predict(input_tensor)
        else:
            pred = self.model(input_tensor)
        
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        acc = sklearn.metrics.accuracy_score(labels, pred)

        if save:

            json_path = self.result_json_path if self.result_json_path is not None else f"data/concepts/{self.concept_filter}/concept_results.json"

            new_row = {
                    "concept": self.concept_filter,
                    "diff_step": diff_step,
                    "pos_gen_audio_samples": pos_tensor.shape[0],
                    "neg_gen_audio_samples": neg_tensor.shape[0], 
                    "embed_model": embed_model,
                    "class_model_type": self.hparams["model_type"],
                    "class_model_train_acc": self.concept_train_accuracy,
                    "class_model_test_acc": self.concept_test_accuracy,
                    "acc": acc,
                    "tpr": tpr,
                    "tnr": tnr,
                }

            if os.path.exists(json_path):
                df = pd.read_json(json_path)
            else:
                df = pd.DataFrame(columns=new_row.keys())

            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            records = df.to_dict(orient="records")
            with open(json_path, "w") as fp:
                json.dump(records, fp, indent=2)

        return acc, tpr, tnr
    

    def _train_svm(self, embeds: np.array, labels: np.array):
        svm_kernel = self.hparams.get("svm_kernel", "rbf")
        svm_C = self.hparams.get("svm_C", 1.0)

        model = sklearn.svm.SVC(kernel=svm_kernel, C=svm_C)
        test_size = self.hparams.get("test_size", 0.2)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            embeds, labels, test_size=test_size
        )

        model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        test_pred  = model.predict(x_test)
        train_acc  = sklearn.metrics.accuracy_score(y_train, train_pred)
        test_acc   = sklearn.metrics.accuracy_score(y_test,  test_pred)

        return train_acc, test_acc, model
    

    def _train_mlp(self, embeds: np.array, labels: np.array):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SimpleMLP(self.hparams)
        test_size = self.hparams["test_size"] if "test_size" in self.hparams else 0.2

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(embeds, labels, test_size=test_size)
        x_train, x_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
        y_train, y_test = torch.tensor(y_train, dtype=torch.int64), torch.tensor(y_test, dtype=torch.int64)

        train_ds = TensorDataset(x_train, y_train)
        test_ds = TensorDataset(x_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=self.hparams["batch_size"], shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=self.hparams["batch_size"])

        model.fit(train_loader, epochs=int(self.hparams["epochs"]), lr=float(self.hparams["lr"]))
        
        accuracy = Accuracy(task="binary")

        train_acc = accuracy(y_train.to(device), model(x_train).to(device)).item()
        test_acc = accuracy(y_test.to(device), model(x_test).to(device)).item()

        return train_acc, test_acc, model

    def _train_logistic_model(self, embeds: np.array, labels: np.array):
        lin_model = sklearn.linear_model.LogisticRegression()
        test_size = self.hparams["test_size"] if "test_size" in self.hparams else 0.2

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(embeds, labels, test_size=test_size)   
        lin_model.fit(x_train, y_train) 

        train_pred, test_pred = lin_model.predict(x_train), lin_model.predict(x_test)

        train_acc = sklearn.metrics.accuracy_score(y_train, train_pred)
        test_acc = sklearn.metrics.accuracy_score(y_test, test_pred)

        return train_acc, test_acc, lin_model

    def _load_concept_embeds(self, pt_path: str, csv_path: str, model_name: str,) -> Tuple[np.array, np.array]:
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Concept embeddings file not found at {pt_path}")

        concept_embeds = torch.load(pt_path)

        batch_size, embed_dim = concept_embeds.shape

        concept_csv = pd.read_csv(csv_path)

        # Tests for corruption / mismatch
        assert len(concept_csv) == batch_size, f"Batch size mismatch between concept embeddings and concept CSV stored at {csv_path}"
        
        if self.model_name == "laion-clap":
            assert embed_dim == 512, f"Embedding dimension mismatch for {model_name} model. Expected 512, got {embed_dim}"
        elif self.model_name == "muq":
            assert embed_dim == 512, f"Embedding dimension mismatch for {model_name} model. Expected 512, got {embed_dim}"

        # Assumes CSV, pt both sorted by positive, negative
        labels = np.concatenate([np.ones(self.num_pos_samples), np.zeros(self.num_neg_samples)])
        concept_embeds_np = concept_embeds.numpy()
        return concept_embeds_np, labels
        
    def _load_gen_audio_embeds(self,
                               diff_step: int, 
                               embed_model: str,
                               pt_path: Optional[None],  
                               csv_path: Optional[str],
                               ) -> Tuple[torch.Tensor]:
        
        # Processing the csv
        csv_path = csv_path if csv_path is not None else "data/generated/audio_info.csv"
        gen_audio_csv = pd.read_csv(csv_path)

        refined_csv = gen_audio_csv[["diffusion_step", "prompt_index", "sample_index", "tag.aspects"]]
        
        concept_filter_name = f"gen_audio_{self.concept_filter}"
        concept_padder_name = f"gen_audio_not_{self.concept_filter}_like"

        pt_path = pt_path if pt_path is not None else f"data/generated/diff_step_{diff_step}/{embed_model}_embeddings.pt"

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Generated audio embeddings file not found at {pt_path}")
        
        gen_audio_embeds = torch.load(pt_path)
        prompts, batch_size, embed_dim = gen_audio_embeds.shape

        assert prompts == 1000, f"{prompts} prompts found; 1000 expected."
        assert batch_size == 7, f"Batch size of {batch_size} found; batch size of 7 expected."

        col_dict, get_true, logic = GEN_AUDIO_FILTER_DICT[self.concept_filter]
        concept_filter_func = create_concept_filter(col_dict, get_true, logic)

        col_dict, get_true, logic = GEN_AUDIO_PADDER_DICT[self.concept_padder]
        concept_padder_func = create_concept_filter(col_dict, get_true, logic)

        pos_samps = concept_filter_func(refined_csv)
        neg_samps = concept_padder_func(refined_csv)

        pos_samps["prompt_index"] = pd.to_numeric(
            pos_samps["prompt_index"], errors="coerce"
        )
        neg_samps["prompt_index"] = pd.to_numeric(
            neg_samps["prompt_index"], errors="coerce"
        )

        pos_indices = torch.tensor(pos_samps["prompt_index"].dropna().astype(int).unique(), dtype=torch.int64)
        neg_indices = torch.tensor(neg_samps["prompt_index"].dropna().astype(int).unique(), dtype=torch.int64)

        num_pos_prompts, num_neg_prompts = len(pos_indices), len(neg_indices)

        pos_tensor = gen_audio_embeds[pos_indices, :, :].reshape(num_pos_prompts * batch_size, embed_dim)
        neg_tensor = gen_audio_embeds[neg_indices, :, :].reshape(num_neg_prompts * batch_size, embed_dim)

        return (pos_tensor, neg_tensor)
    
    def get_pca(self, diff_step: int, embed_model: str, resolution: int = 200,):
        # In future change this
        pt_path = f"{self.dir_path}/{embed_model}_embeddings.pt"
        concept_embeds, concept_labels = self._load_concept_embeds(csv_path=f"{self.dir_path}/audio_info.csv", 
                                                                   model_name=embed_model,
                                                                   pt_path=pt_path)
        pos_tensor, neg_tensor = self._load_gen_audio_embeds(diff_step=diff_step, embed_model=embed_model, pt_path=f"data/generated/diff_step_{diff_step}/{embed_model}_embeddings.pt", csv_path=f"data/generated/audio_info.csv")
        gen_audio_embeds = torch.concat([pos_tensor, neg_tensor]).numpy()
        gen_audio_labels = np.concatenate([np.ones(pos_tensor.shape[0]), np.zeros(neg_tensor.shape[0])])

        pca = PCA(n_components=2)
        train2d = pca.fit_transform(concept_embeds)
        test2d = pca.transform(gen_audio_embeds)

        x0_min, x0_max = train2d[:,0].min()-1, train2d[:,0].max()+1
        x1_min, x1_max = train2d[:,1].min()-1, train2d[:,1].max()+1
        xx, yy = np.meshgrid(
            np.linspace(x0_min, x0_max, resolution),
            np.linspace(x1_min, x1_max, resolution)
        )
        grid2d = np.c_[xx.ravel(), yy.ravel()]

        # Computing predictions
        grid_orig = pca.inverse_transform(grid2d)
        if self.hparams["model_type"] in ("logistic", "svm"):
            Z = self.model.predict(grid_orig)
        else:  # mlp
            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.tensor(grid_orig, dtype=torch.float32))
        Z = Z.reshape(xx.shape)

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 8))

        model_type = self.hparams["model_type"]

        ax.contourf(xx, yy, Z, alpha=0.2)
        ax.scatter(train2d[concept_labels==1,0], train2d[concept_labels==1,1],
               marker="o", edgecolor="k", facecolor="tab:green", label="Concept +")
        ax.scatter(train2d[concept_labels==0,0], train2d[concept_labels==0,1],
               marker="x", edgecolor="k", facecolor="tab:red", label="Concept -")
        ax.scatter(test2d[gen_audio_labels==1,0], test2d[gen_audio_labels==1,1],
               marker="o", edgecolor="tab:green", facecolor="none", label="Diffusion Audio +")
        ax.scatter(test2d[gen_audio_labels==0,0], test2d[gen_audio_labels==0,1],
                marker="x", edgecolor="tab:red",  facecolor="none", label="Diffusion Audio -")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{model_type} decision boundary on PCA projection")
        ax.legend(loc="best")
        plt.tight_layout()
        return fig