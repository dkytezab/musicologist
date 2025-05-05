from typing import Optional, List, Dict, Tuple
import torch
import os
import pandas as pd
import numpy as np
import sklearn.linear_model
from sklearn.metrics import recall_score, confusion_matrix

from concept_filters import load_concept_filter

class BinaryClassifier(object):
    def __init__(self, 
                 concept_filter: str, 
                 num_pos_samples: int,
                 num_neg_samples: int,
                 model_name: str,
                 concept_pt_path: Optional[str] = None,
                 dir_path: Optional[str] = None,
                 hparams: Optional[Dict] = None,
                 ):

        self.concept_filter = concept_filter
        self.model_name = model_name

        self.dir_path = dir_path if dir_path is not None else f"data/concepts/{concept_filter}"  
        self.concept_pt_path = concept_pt_path if concept_pt_path is not None else f"{self.dir_path}/{model_name}_embeds.pt"
        self.hparams = hparams if hparams is not None else {}

        self.num_pos_samples = num_pos_samples
        self.num_neg_samples = num_neg_samples
        
    def train(self, csv_path: Optional[str] = None) -> None:
        csv_path = csv_path if csv_path is not None else f"{self.dir_path}/audio_info.csv"
        
        embeds, labels = self._load_concept_embeds(self.concept_pt_path, csv_path, self.model_name)

        if self.hparams["model_type"] == "logistic":
            tr_acc, te_acc, lin_model = self._train_logistic_model(embeds, labels)
        else:
            raise NotImplementedError(f"Model type {self.hparams['model_type']} not implemented")

        self.concept_train_accuracy = tr_acc
        self.concept_test_accuracy = te_acc
        self.model = lin_model
    
    def inference(self, pt_path: str,) -> None:
        pos_tensor, neg_tensor = self._load_gen_audio_embeds(pt_path=pt_path, csv_path=None)
        labels = np.concatenate([np.ones(pos_tensor.shape[0]), np.zeros(neg_tensor.shape[0])])
        input_tensor = torch.concat([pos_tensor, neg_tensor])

        pred = self.model.predict(input_tensor)
        tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        acc = sklearn.metrics.accuracy_score(labels, pred)

        return acc, tpr, tnr

    def _train_logistic_model(self, embeds: np.array, labels: np.array) -> None:
        lin_model = sklearn.linear_model.LogisticRegression()
        test_size = self.hparams["test_size"] if "test_size" in self.hparams else 0.2

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(embeds, labels, test_size=test_size)   
        lin_model.fit(x_train, y_train) 

        train_pred, test_pred = lin_model.predict(x_train), lin_model.predict(x_test)

        train_acc = sklearn.metrics.accuracy_score(y_train, train_pred)
        test_acc = sklearn.metrics.accuracy_score(y_test, test_pred)

        return train_acc, test_acc, lin_model

    def _load_concept_embeds(self, pt_path: str, csv_path: str, model_name: str,) -> torch.Tensor:
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
            assert embed_dim == 1024, f"Embedding dimension mismatch for {model_name} model. Expected 1024, got {embed_dim}"

        # Assumes CSV, pt both sorted by positive, negative
        labels = np.concatenate([np.ones(self.num_pos_samples), np.zeros(self.num_neg_samples)])
        concept_embeds_np = concept_embeds.numpy()
        return concept_embeds_np, labels
        

    def _load_gen_audio_embeds(self, 
                               pt_path: str,  
                               csv_path: Optional[str],
                               ) -> Tuple[torch.Tensor]:
        
        # Processing the csv
        csv_path = csv_path if csv_path is not None else "data/generated/audio_info.csv"
        gen_audio_csv = pd.read_csv(csv_path)

        refined_csv = gen_audio_csv[["diffusion_step", "prompt_index", "sample_index", "tag.aspects"]]
        
        concept_filter_name = f"gen_audio_{self.concept_filter}"
        concept_padder_name = f"gen_audio_not_{self.concept_filter}_like"

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Generated audio embeddings file not found at {pt_path}")
        
        gen_audio_embeds = torch.load(pt_path)
        prompts, batch_size, embed_dim = gen_audio_embeds.shape

        assert prompts == 1000, f"{prompts} prompts found; 1000 expected."
        assert batch_size == 7, f"Batch size of {batch_size} found; batch size of 7 expected."

        concept_filter_func = load_concept_filter(func_name=concept_filter_name)
        concept_padder_func = load_concept_filter(func_name=concept_padder_name)

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

